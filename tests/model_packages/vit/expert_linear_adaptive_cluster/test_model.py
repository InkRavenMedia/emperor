import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from emperor.datasets.image.classification import JointImageClassification, Mnist
from emperor.neuron import NeuronClusterMonitorCallback
from emperor.transformer import SequenceExpansion, TransformerEncoderLayerState
from models.catalog import model_package
from models.project_cli.main import run_experiment
from tests.support.dataset_metadata import offline_dataset_metadata

BASE = model_package("vit/expert_linear_adaptive")
CLUSTER = model_package("vit/expert_linear_adaptive_cluster")
SMALL = {"image_patch_size": 14, "stack_num_layers": 1}


def model(package=CLUSTER, preset="baseline", **overrides):
    values = package.runtime_defaults_spec.parse_overrides({**SMALL, **overrides})
    return package.build_model(
        package.build_configuration(
            package.resolve_preset(preset), Mnist, config_overrides=values
        )
    ).eval()


def split(output):
    return output if isinstance(output, tuple) else (output, None)


def counts(trace):
    rounds = [
        trace.entry_selected_coordinates,
        *(step.selected_coordinates for step in trace.steps),
    ]
    return [coords[:, 0, 0].tolist() for coords in rounds]


def test_inherits_vit_options_presets_and_tasks():
    defaults = CLUSTER.runtime_defaults_spec
    for key, value in BASE.runtime_defaults_spec.default_items():
        assert defaults.serialize_value(
            defaults.current_value(key)
        ) == defaults.serialize_value(value), key
    assert {p.name for p in CLUSTER.preset_type} == {
        p.name for p in BASE.preset_type
    } | {"THREE_NEURON_CHAIN"}
    assert CLUSTER.dataset_metadata == BASE.dataset_metadata
    assert {"CLUSTER_ROUTING_MODE", "CLUSTER_ATTENTION_RESIDUAL_FLAG"} <= set(
        defaults.ordered_configuration_keys()
    )


@pytest.mark.parametrize(
    "preset",
    [
        "baseline",
        "post-norm",
        "sinusoidal",
        "attention-bias",
        "low-rank-expert-weight",
        "top1-switch-aux",
    ],
)
def test_single_neuron_matches_the_original_vit_with_identical_weights(preset):
    original = model(BASE, preset)
    clustered = model(CLUSTER, preset, cluster_x_axis_total_neurons=1)
    for name in ("patch", "positional_embedding", "encoder_layer_norm", "output"):
        getattr(clustered, name).load_state_dict(getattr(original, name).state_dict())
    nucleus = clustered.transformer.cluster["neuron_1_1_1"].nucleus
    nucleus.model.load_state_dict(original.transformer.state_dict())
    images = torch.randn(2, 1, 28, 28)
    torch.manual_seed(123)
    original_logits, original_aux = split(original(images))
    torch.manual_seed(123)
    logits, aux = split(clustered(images))
    torch.testing.assert_close(logits, original_logits)
    torch.testing.assert_close(aux, original_aux)


def test_chain_matches_direct_composition_and_checkpoint():
    classifier = model(preset="three-neuron-chain")
    cluster = classifier.transformer
    neurons = list(cluster.cluster.values())
    assert all(neuron.sequence_residual is None for neuron in neurons)
    parameters = [next(neuron.nucleus.parameters()) for neuron in neurons]
    assert len({p.data_ptr() for p in parameters}) == 3
    tokens = torch.randn(2, 5, 32, requires_grad=True)
    torch.manual_seed(53)
    result, trace = cluster(
        TransformerEncoderLayerState(hidden=tokens), return_trace=True
    )
    assert counts(trace) == [[1, 1], [2, 2], [3, 3]]
    torch.manual_seed(53)
    expected = TransformerEncoderLayerState(hidden=tokens)
    losses = []
    for neuron in neurons:
        expected = neuron.nucleus(TransformerEncoderLayerState(hidden=expected.hidden))
        if expected.loss is not None:
            losses.append(expected.loss)
    torch.testing.assert_close(result.hidden, expected.hidden)
    torch.testing.assert_close(result.loss, sum(losses))
    torch.manual_seed(53)
    ordinary_chain = nn.Sequential(*neurons)(
        TransformerEncoderLayerState(hidden=tokens)
    )
    torch.testing.assert_close(result.hidden, ordinary_chain.hidden)
    torch.testing.assert_close(result.loss, ordinary_chain.loss)
    (result.hidden.square().sum() + result.loss).backward()
    assert all(parameter.grad is not None for parameter in parameters)
    restored = model(preset="three-neuron-chain")
    restored.load_state_dict(classifier.state_dict(), strict=True)
    torch.manual_seed(53)
    torch.testing.assert_close(
        restored.transformer(TransformerEncoderLayerState(hidden=tokens)).hidden,
        result.hidden,
    )


def test_internal_recurrence_and_history_start_fresh_at_each_visit():
    classifier = model(
        preset="three-neuron-chain",
        cluster_attention_residual_flag=True,
        recurrent_flag=True,
        recurrent_max_steps=2,
        recurrent_initial_iterations=2,
        recurrent_residual_connection_option="AttentionResidualConfig",
        recurrent_residual_block_size=1,
        recurrent_residual_rms_norm_epsilon=1e-6,
        recurrent_stack_halting_flag=True,
        recurrent_min_steps=2,
    )
    neurons = list(classifier.transformer.cluster.values())
    visits, passes, residual_visits = [], [], []

    def record_residual_visit(module, args, kwargs, output):
        assert set(kwargs) == {"residual_state"}
        residual_visits.append(module)

    for index, neuron in enumerate(neurons):
        neuron.sequence_residual.register_forward_hook(
            record_residual_visit, with_kwargs=True
        )
        neuron.nucleus.register_forward_pre_hook(
            lambda _, args, i=index: visits.append(
                (i, args[0].halting_state, args[0].residual_state)
            )
        )
        neuron.nucleus.model.block_model.register_forward_hook(
            lambda block, args, result, i=index: passes.append((i, id(block)))
        )
    hidden = torch.randn(2, 5, 32)
    output = classifier.transformer(TransformerEncoderLayerState(hidden=hidden))
    assert residual_visits == [neuron.sequence_residual for neuron in neurons]
    assert visits == [(0, None, None), (1, None, None), (2, None, None)]
    for index in range(3):
        ids = [block for i, block in passes if i == index]
        assert len(ids) == 2 and len(set(ids)) == 1
    output.hidden.square().sum().backward()
    assert all(neuron.sequence_residual.query.grad is not None for neuron in neurons)


class ImageSampler(nn.Module):
    """Use an image-dependent entry decision or force known neighboring targets."""

    def __init__(self, select):
        super().__init__()
        self.select = select

    def sample_probabilities_and_indices(self, summary):
        indices, weights = self.select(summary)
        return weights, indices, None, summary.new_zeros(())


def fixed_sampler(indices, weights):
    return ImageSampler(
        lambda summary: (
            torch.tensor(indices, device=summary.device).expand(summary.size(0), -1),
            summary.new_tensor(weights).expand(summary.size(0), -1),
        )
    )


def test_learned_routes_keep_each_image_together_and_can_halt_independently(
    monkeypatch,
):
    classifier = model(
        cluster_routing_mode="LEARNED",
        cluster_x_axis_total_neurons=3,
        cluster_halting_flag=True,
        cluster_halting_threshold=0.9,
    )
    cluster = classifier.transformer
    widths = []
    cluster.entry_sampler = ImageSampler(
        lambda summary: ((summary[:, :1] < 0).long(), torch.ones_like(summary[:, :1]))
    )
    for neuron in cluster.cluster.values():
        # Keep the route signal legible to the real stick-breaking controller.
        def identity(state):
            widths.append(state.hidden.shape[1])
            return state

        monkeypatch.setattr(neuron.nucleus.model, "forward", identity)
        own = neuron.terminal.neuron_connections
        next_index = torch.where((own == own.new_tensor([3, 1, 1])).all(dim=1))[0]
        if next_index.numel():
            neuron.terminal.sampler = fixed_sampler([int(next_index[0])], [1.0])

    def gates(summary):
        return torch.stack((-summary[:, 0] * 8, summary[:, 0] * 8), dim=1)

    monkeypatch.setattr(
        cluster.halting_model.halting_gate_model, "forward_hidden", gates
    )
    hidden = torch.stack((torch.ones(5, 32), -torch.ones(5, 32))).requires_grad_()
    result, trace = cluster(
        TransformerEncoderLayerState(hidden=hidden), return_trace=True
    )
    assert trace.entry_selected_coordinates[:, 0, 0].tolist() == [1, 2]
    assert trace.entry_halt_mask.tolist() == [True, False]
    assert trace.steps[0].valid_mask[:, 0].tolist() == [False, True]
    assert trace.steps[0].selected_coordinates[1, 0, 0].item() == 3
    assert widths == [5, 5, 5, 5]
    torch.testing.assert_close(result.hidden, hidden, atol=1e-5, rtol=1e-5)
    (result.hidden.sum() + result.loss).backward()
    assert torch.isfinite(hidden.grad).all()


def test_top_k_expansion_and_residual_merge_uses_only_valid_slots():
    classifier = model(
        cluster_routing_mode="LEARNED",
        cluster_x_axis_total_neurons=3,
        cluster_terminal_top_k=2,
        cluster_max_steps=1,
        cluster_attention_residual_flag=True,
        sequence_expansion_flag=True,
        sequence_expansion_max_new_tokens=2,
        sequence_expansion_model_option="LINEAR",
        sequence_expansion_model_bias_flag=True,
    )
    cluster = classifier.transformer
    first, second, third = cluster.cluster.values()
    cluster.entry_sampler = fixed_sampler([0, 1], [0.75, 0.25])
    branch_outputs = []
    for neuron, expand in ((first, False), (second, True), (third, False)):
        expansion = next(
            module
            for module in neuron.modules()
            if isinstance(module, SequenceExpansion)
        )
        with torch.no_grad():
            expansion.count_predictor.weight_params.zero_()
            expansion.count_predictor.bias_params.copy_(
                torch.tensor([0.0, 2.0]) if expand else torch.tensor([2.0, 0.0])
            )
        neuron.nucleus.register_forward_hook(
            lambda _, args, out: branch_outputs.append(out)
        )
    # Both entry branches execute. The winner's terminal routes to B on round two.
    destination = torch.where(
        (first.terminal.neuron_connections == torch.tensor([2, 1, 1])).all(dim=1)
    )[0].item()
    first.terminal.sampler = fixed_sampler([destination], [1.0])
    received = []
    second.nucleus.register_forward_pre_hook(
        lambda _, args: received.append(replace(args[0]))
    )
    hidden = torch.randn(2, 4, 32, requires_grad=True)
    padding = torch.tensor([[False, False, False, False], [False, False, False, True]])
    result, trace = cluster(
        TransformerEncoderLayerState(hidden=hidden, key_padding_mask=padding),
        return_trace=True,
    )
    assert len(branch_outputs) == 3 and len(received) == 2
    assert trace.entry_chosen_branch_indices.tolist() == [0, 0]
    assert trace.steps[0].selected_coordinates[:, 0, 0].tolist() == [2, 2]
    assert received[1].key_padding_mask.tolist() == [[False] * 6, [False] * 5 + [True]]
    # At zero-initialized depth queries, old positions average input and
    # current output. A new slot exists only on B, with one available source.
    torch.testing.assert_close(
        received[1].hidden[0, 5], 0.25 * branch_outputs[1].hidden[0, 5]
    )
    expected_existing = (
        hidden[:, 0]
        + 0.75 * branch_outputs[0].hidden[:, 0]
        + 0.25 * branch_outputs[1].hidden[:, 0]
    ) / 2
    torch.testing.assert_close(received[1].hidden[:, 0], expected_existing)
    assert result.key_padding_mask.tolist() == [[False] * 8, [False] * 7 + [True]]
    (result.hidden[:, 0].square().sum() + result.loss).backward()
    assert second.sequence_residual.query.grad is not None
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("dataset", [Mnist, JointImageClassification])
def test_learned_growth_and_pruning_sync_the_optimizer_in_each_task(
    dataset, monkeypatch
):
    joint = dataset is JointImageClassification
    patch = 16 if joint else 14
    overrides = CLUSTER.runtime_defaults_spec.parse_overrides(
        {
            "image_patch_size": patch,
            "stack_num_layers": 1,
            "cluster_routing_mode": "LEARNED",
            "cluster_terminal_top_k": 2,
            "cluster_initial_x_axis_total_neurons": 1,
            "cluster_x_axis_total_neurons": 2,
            "cluster_growth_threshold": 1,
            "cluster_pruning_threshold": 1,
            "cluster_growth_warmup_steps": 2,
            "cluster_attention_residual_flag": True,
        }
    )
    config = CLUSTER.build_configuration(dataset=dataset, config_overrides=overrides)
    classifier = CLUSTER.build_model(config)
    cluster = (classifier.encoder if joint else classifier).transformer
    optimizer = classifier.configure_optimizers()
    old_names = set(cluster.cluster)
    inputs = (
        (torch.randn(2, 3, 32, 32), "cifar100")
        if joint
        else (torch.randn(2, 1, 28, 28),)
    )
    logits, aux = split(classifier(*inputs))
    assert logits.shape == (2, 100 if joint else 10)
    (logits.square().mean() + aux).backward()
    assert any(
        parameter.grad is not None
        for parameter in cluster.cluster[next(iter(old_names))].nucleus.parameters()
    )
    (grown,) = set(cluster.cluster) - old_names
    grown_parameters = {
        id(parameter) for parameter in cluster.cluster[grown].parameters()
    }
    assert grown_parameters.isdisjoint(
        {id(p) for group in optimizer.param_groups for p in group["params"]}
    )
    callback = CLUSTER.runtime_defaults_spec.current_value(
        "CALLBACK_NEURON_CLUSTER_OPTIMIZER_SYNC"
    )
    callback.sync_optimizers(SimpleNamespace(optimizers=[optimizer]), classifier)
    assert grown_parameters <= {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    restored = CLUSTER.build_model(config)
    restored.load_state_dict(classifier.state_dict(), strict=True)
    classifier.eval()
    restored.eval()
    torch.manual_seed(45)
    expected = classifier(*inputs)
    torch.manual_seed(45)
    torch.testing.assert_close(restored(*inputs), expected)

    # Only the original neuron is on the entry plane. Route off-grid on the
    # next batch so the newly grown neuron is eligible for actual pruning.
    terminal = cluster.cluster[next(iter(old_names))].terminal
    escape = torch.where(terminal.neuron_connections[:, 2] == 0)[0][0].item()
    monkeypatch.setattr(
        terminal.sampler,
        "sample_probabilities_and_indices",
        fixed_sampler([escape], [1.0]).sample_probabilities_and_indices,
    )
    classifier.train()
    classifier(*inputs)
    assert set(cluster.cluster) == old_names
    callback.sync_optimizers(SimpleNamespace(optimizers=[optimizer]), classifier)
    assert grown_parameters.isdisjoint(
        {id(p) for group in optimizer.param_groups for p in group["params"]}
    )


def test_cluster_monitor_preserves_encoder_output_and_optional_trace():
    classifier = model(preset="three-neuron-chain").train()
    monitor = NeuronClusterMonitorCallback(log_every_n_steps=1)
    monitor.on_fit_start(None, classifier)
    state = TransformerEncoderLayerState(hidden=torch.randn(2, 5, 32))
    result = classifier.transformer(state)
    assert result.hidden.shape == state.hidden.shape
    _, trace = classifier.transformer(state, return_trace=True)
    assert counts(trace) == [[1, 1], [2, 2], [3, 3]]
    monitor.on_fit_end(None, classifier)


def test_learned_branches_receive_their_images_attention_heads_and_padding():
    classifier = model(cluster_routing_mode="LEARNED", cluster_x_axis_total_neurons=2)
    cluster = classifier.transformer
    cluster.entry_sampler = ImageSampler(
        lambda pooled: ((pooled[:, :1] < 0).long(), torch.ones_like(pooled[:, :1]))
    )
    seen = []
    for neuron in cluster.cluster.values():
        neuron.nucleus.register_forward_pre_hook(
            lambda _, args: seen.append(replace(args[0]))
        )
        outside = torch.where(neuron.terminal.neuron_connections[:, 2] == 0)[0][
            0
        ].item()
        neuron.terminal.sampler = fixed_sampler([outside], [1.0])
    hidden = torch.stack([torch.ones(5, 32), -torch.ones(5, 32)])
    padding = torch.tensor(
        [[False, True, False, False, False], [False, False, False, True, False]]
    )
    masks = torch.zeros(2, 4, 5, 5, dtype=torch.bool)
    masks[0, :, :, 2] = True
    masks[1, :, :, 1] = True
    result = cluster(
        TransformerEncoderLayerState(
            hidden=hidden, key_padding_mask=padding, attention_mask=masks.flatten(0, 1)
        )
    )
    assert len(seen) == 2
    for index, incoming in enumerate(seen):
        torch.testing.assert_close(incoming.attention_mask, masks[index])
        torch.testing.assert_close(
            incoming.key_padding_mask, padding[index : index + 1]
        )
    torch.testing.assert_close(result.key_padding_mask, padding)
    torch.testing.assert_close(result.attention_mask, masks.flatten(0, 1))


@pytest.mark.training
@pytest.mark.parametrize("dataset", [Mnist, JointImageClassification])
def test_documented_cli_runs_checkpoint_the_cluster(tmp_path, monkeypatch, dataset):
    joint = dataset is JointImageClassification
    selection = (
        ["--preset", "baseline", "--experiment-task", "joint-image-classification"]
        if joint
        else ["--preset", "three-neuron-chain", "--datasets", "mnist"]
    )
    config = [
        "--trainer-enable-checkpointing",
        "true",
        "--callback-checkpoint-flag",
        "true",
        "--num-epochs",
        "1",
        "--batch-size",
        "16",
        "--image-patch-size",
        "16" if joint else "14",
        "--data-num-workers",
        "0",
        "--seed",
        "17",
    ]
    if joint:
        config.extend(
            [
                "--cluster-routing-mode",
                "LEARNED",
                "--cluster-x-axis-total-neurons",
                "3",
                "--cluster-terminal-top-k",
                "2",
                "--cluster-attention-residual-flag",
                "true",
                "--stack-num-layers",
                "1",
            ]
        )
    monkeypatch.chdir(tmp_path)
    with offline_dataset_metadata(dataset, tmp_path, seed=17):
        assert (
            run_experiment(
                [
                    "--model-type",
                    "vit",
                    "--model",
                    "expert_linear_adaptive_cluster",
                    *selection,
                    "--config",
                    *config,
                ]
            )
            == 0
        )
    (receipt,) = (tmp_path / "logs").rglob("result.json")
    result = json.loads(receipt.read_text())
    assert result["experimentTask"] == (
        "joint-image-classification" if joint else "image-classification"
    )
    checkpoint = torch.load(
        receipt.parent / "checkpoints" / "last.ckpt", weights_only=True
    )
    assert checkpoint["global_step"] > 0 and len(checkpoint["optimizer_states"]) == 1
    state = checkpoint["state_dict"]
    for x in range(1, 4):
        assert any(
            f"transformer.cluster.neuron_{x}_1_1.nucleus." in key for key in state
        )
    if joint:
        assert all(
            f"test/{name}/accuracy" in result["metrics"]
            for name in JointImageClassification.sources
        )
