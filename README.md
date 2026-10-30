# Emperor

**Current package version:** `0.1.0`

Emperor is a personal PyTorch research framework for building experimental neural
network models from reusable, inspectable components. The repository combines:

- `src/emperor/` - core neural modules, controller primitives, datasets, training
  helpers, and monitor callbacks.
- `src/model_runtime/` - consumer-neutral Model Package, Inspection, Run, and
  project Adapter interfaces shared by the CLI and Workbench.
- `src/models/` - reference experiment packages that compose the core modules into
  runnable architectures.
- `apps/workbench/api/` - the separately installable FastAPI application for
  inspection, run history, config snapshots, and local training jobs.
- `apps/workbench/web/` - the Next.js Workbench frontend for planning experiments
  and reviewing live or historical monitor data.

The main goal is repeatable model experimentation: build small models, compare
variants, watch what happens during training, and use the results to decide
which ideas are worth exploring next.

## Environment Setup

**System requirements:** Git and [mise](https://mise.jdx.dev/). Mise provisions
the pinned Python 3.13 and Node 24 toolchains; Visual Studio, Xcode, and a
system Python or Node installation are not setup prerequisites.

The guaranteed CPU baseline covers Ubuntu 24.04 x64, WSL2 Ubuntu 24.04 x64,
Windows 11 x64 with PowerShell, and macOS 15+ Apple Silicon. Intel macOS is not
part of the baseline because PyTorch 2.12 has no Intel macOS wheel for the
required Python 3.13 runtime. MPS execution and CUDA on platforms other than
Linux x86-64 remain outside the guaranteed baseline.

Set up the repository and start the Workbench with the same commands everywhere:

```text
mise run setup --profile cpu
mise run dev
```

Setup creates `./torchenv` with the native `bin` or `Scripts` layout, installs
binary Python dependencies from the matching platform lock, and runs `npm ci`
for the locked root development tools and Workbench frontend. It records a hash
of every setup input. Repeating setup is idempotent. Switching profiles
recreates the launcher-owned virtualenv. CUDA profiles are available only on
Linux x86-64. The current profile uses PyTorch 2.12.0's CUDA 13.0 wheels:

```text
mise run setup --profile cuda
mise run dev --profile cuda
```

The `cuda-legacy` profile uses the official CUDA 12.6 wheels and guarantees the
GTX 1080 Ti (`sm_61`) target:

```text
mise run setup --profile cuda-legacy
mise run dev --profile cuda-legacy
```

Always pass the selected non-CPU profile to `dev`; its default remains `cpu`.
Other GPUs supported by the CUDA 12.6 wheel are best-effort. The wheels supply
the CUDA runtime, so a system CUDA toolkit is not required. The host NVIDIA
driver remains external: CUDA 12.x minor compatibility requires Linux driver
[525.60.13 or newer](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
while driver
[560.35.05 or newer](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html)
avoids minor-compatibility feature limits for the pinned CUDA 12.6 Update 3
runtime. The CUDA 13 profile requires an R580-or-newer driver.

The native locks live under
`constraints/python-3.13-<platform>-<arch>-<profile>.txt`. The legacy lock is
standalone; it does not inherit any CUDA 13 pins.

### GTX 1080 Ti hardware gate

Setup verifies the Torch version, CUDA runtime, and compatible compiled Pascal
code without requiring an attached GPU. PyTorch 2.12's CUDA 12.6 Linux wheel
reports an `sm_60` cubin; NVIDIA's same-major forward binary compatibility
[makes that cubin valid](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#binary-compatibility)
on the GTX 1080 Ti's `sm_61` capability. Run this manual gate on the target
machine before treating the hardware path as validated.

First confirm that the driver sees the card:

```bash
nvidia-smi
```

Then verify the exact build and physical device capability:

```bash
mise exec -- python tools/emperor_dev.py python -- -c \
  "import torch; assert torch.__version__ == '2.12.0+cu126'; assert torch.version.cuda == '12.6'; assert torch.cuda.get_device_capability(0) == (6, 1)"
```

Allocate and compute a CUDA tensor:

```bash
mise exec -- python tools/emperor_dev.py python -- -c \
  "import torch; x = torch.arange(1024, device='cuda'); print((x * x).sum().item())"
```

Finally run one explicit-GPU training epoch:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config \
  --num-epochs 1 \
  --trainer-accelerator gpu \
  --trainer-devices 1
```

Emperor does not automatically select a GPU or change training precision for
this profile.

The Workbench is available at:

```text
http://localhost:9000
```

The local Workbench API defaults to:

```text
http://127.0.0.1:9999
```

Start, inspect, or stop validated Workbench process trees:

```text
mise run workbench:start
mise run workbench:status
mise run workbench:stop
```

Use the `mise` tasks shown above on Unix-like systems. PowerShell users can run
`. .\env.ps1`, `. .\env.ps1 -WorkbenchStatus`, or
`. .\env.ps1 -WorkbenchStop` to activate the virtualenv and manage Workbench.

Default ports can be overridden with:

```bash
export WORKBENCH_BACKEND_PORT=9999
export WORKBENCH_FRONTEND_PORT=9000
export NEXT_PUBLIC_WORKBENCH_API_URL=http://127.0.0.1:9999
```

## Current State

Use this command to validate the full local setup:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline
```

`linears/linear` is the recommended smoke-test model. It currently exposes
baseline, gating, halting, memory, residual, post-norm, and recurrent preset
families. The other cataloged model packages are research/reference
implementations for adaptive parameters, experts, neuron clusters, and
transformer experiments. Before using one for a long run, verify the exact
preset with `--print-model` and a small one-epoch run.

The Workbench has grown into a local experiment workbench with three workspaces:

- **Model** - choose a model/preset/dataset, inspect module and operation
  graphs, review parameters, create config snapshots, and inspect historical
  training-run targets.
- **Training** - plan, start, and monitor local training jobs for the selected
  model configuration.
- **Logs** - browse historical TensorBoard runs, inspect saved graphs, and
  review metrics and monitor data from completed runs.

## Optional Soft Mixture of Experts

Core mixture configurations can select the noncausal Soft MoE method from
[From Sparse to Soft Mixtures of Experts](https://arxiv.org/pdf/2308.00951v2).
It forms learned weighted token summaries, runs the existing expert networks on
those slots, and combines their outputs back into token positions. The existing
token sampler is unchanged. Leaving `soft_moe_config=None` preserves existing
routing, seeded initialization, and checkpoint keys.

For an existing `MixtureOfExpertsConfig` with a pointwise linear expert stack:

```python
from dataclasses import replace

from emperor.augmentations.adaptive_parameters import DepthMappingHandlerConfig
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
    SoftMoEConfig,
)

soft_config = replace(
    existing_mixture_config,
    soft_moe_config=SoftMoEConfig(
        slots_per_expert=1,
        model_config=DepthMappingHandlerConfig(model_config=scoring_stack_config),
    ),
    top_k=existing_mixture_config.num_experts,
    capacity_factor=0.0,
    dropped_token_behavior=DroppedTokenOptions.ZEROS,
    routing_initialization_mode=RoutingInitializationMode.LAYER,
    sampler_config=None,
    compute_expert_mixture_flag=True,
    weighted_parameters_flag=True,
    weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
)
mixture = soft_config.build()
output, returned_mask, expert_loss = mixture(tokens, skip_mask=active_tokens)
```

Inputs are explicitly `[batch, tokens, features]`; optional `active_tokens` is
Boolean `[batch, tokens, 1]`, with True meaning valid. Slot count is fixed while
token count can vary. Masked values are removed before arithmetic; fully masked
sequences produce zero mixture outputs. All experts process their fixed slots;
`top_k=num_experts` is configuration coherence, with no top-k operation or capacity
dropping. Expert auxiliary losses retain their existing semantics, including
execution on zero slots for fully masked sequences.

The existing `MixtureOfExpertsLayerConfig` / `MixtureOfExpertsModelConfig` path
also supports this option: select LAYER routing, matching `top_k`, and no sampler
on the outer model. A `FeedForwardConfig` directly wrapping that model preserves
sequence boundaries. Transformer blocks transport padding and explicit
`batch_first_flag` layout. Direct feed-forward calls accept `batch_first=False`
for `[tokens, batch, features]` and `padding_mask` of shape `[batch, tokens]`, where
Boolean True or additive negative infinity excludes tokens. The surrounding
Transformer residual retains its ordinary padded-output behavior.

Causal attention and explicit attention masks are rejected: global slot pooling
does not preserve their visibility constraints. Shared routing, attention
Map/Reduce, recurrent wrappers, adaptive grouping, and context controllers are
unsupported in this mode. Existing ordinary linear and compatible per-row
adaptive linear expert stacks are reused without replacing their networks.
This is a core configuration option; package-specific CLI presets are unchanged.

Scoring uses a configurable depth-mapping stack with independent expert parameters
in each layer. Supply an ordinary `LayerStackConfig` as `scoring_stack_config`,
with `apply_output_postprocessing_flag=False` so its final layer returns raw
logits. Inputs use `x / sqrt(sum(x*x) + 1e-6)`, followed by the stack and a learned
scalar initialized to one. This extends the paper's normalized linear scorer.
Slots pool original token values. Scoring inputs and parameter views are promoted
to float32 (float64 for float64 inputs), while eligible operations follow the
caller's autocast setting. Weighted sums use the routing weights' dtype; expert
execution follows normal model dtype/autocast settings. No routing noise or
balancing loss is added.

The independent numerical oracle and integration tests are in
`tests/unit/test_expert_soft_moe.py`. They validate forward values, gradients,
masking, precision, sequence independence, checkpoints, and Transformer wiring.
They do not establish training-quality, GPU, distributed, or performance parity.

## Quick Start

After `mise run dev`, use this path to inspect the stable model and run one
short training job.

```bash
# 1. Confirm the model categories.
mise run experiment -- --list-model-types

# 2. Confirm the available linear models.
mise run experiment -- \
  --model-type linears \
  --list-models

# 3. Confirm the available linear presets.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-presets

# 4. Inspect the baseline model without training.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --print-model

# 5. Check optional monitor callbacks for training runs.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-monitors

# 6. Run a one-epoch smoke training job.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --logdir quickstart \
  --config --num-epochs 1

# 7. Open TensorBoard logs if you want the raw event view.
tensorboard --logdir logs/quickstart
```

The smoke run builds the `baseline` preset, trains it on `mnist` for one epoch,
and may download the dataset on first use.

To train the decoder-only GPT baseline from scratch for one WikiText-2 epoch:

```bash
mise run experiment -- \
  --model-type gpt \
  --model linear \
  --preset baseline \
  --datasets wiki-text2 \
  --logdir gpt-wikitext2 \
  --config --num-epochs 1
```

The default GPT modes use causal next-token loss and return token IDs from greedy
generation. Their datasets are `penn-treebank`, `wiki-text2`, `wiki-text103`, and
`open-web-text`. All four GPT packages (`linear`, `linear_adaptive`,
`expert_linear`, and `expert_linear_adaptive`) also offer the `hierarchical`
preset with `wiki-text103-hierarchical`: a vocabulary-free byte
encoder, GPT backbone, and byte decoder with `generate_text`. See the
[explicit reference configuration](src/emperor/transformer/hierarchical/_reference.py).

All four BERT packages accept `hierarchical_embedding_config` and
`soft_moe_config` independently or together through `config_overrides`.
Supply a `HierarchicalByteEmbeddingConfig` whose output dimension matches BERT's
hidden dimension, and a `SoftMoEConfig` with its depth-mapped scoring stack:

```python
from models.catalog import model_package

package = model_package("bert/expert_linear_adaptive")
configuration = package.build_configuration(
    config_overrides={
        "hierarchical_embedding_config": byte_embedding_config,
        "soft_moe_config": soft_moe_config,
        "mlm_decoder_weight_tying_flag": False,
    }
)
model = package.build_model(configuration)
model.set_token_vocabulary(token_spellings_in_id_order)
```

Trainer setup binds the prepared BERT tokenizer automatically. Vocabulary binding
is saved in checkpoints and preserves exact spellings, including `[MASK]`,
`[CLS]`, `[SEP]`, and WordPiece prefixes. This embedding mode retains BERT's
vocabulary-based masked-token and next-sentence heads, sentence positions, and
segment embeddings. Its MLM head must be untied from the input embedding.
Soft MoE replaces the feed-forward routing; attention retains the package's
configured implementation unless `attention_soft_moe_config` is also supplied.
It requires bidirectional attention and pointwise experts without shared
adaptive grouping, recurrent expert stacks, or context controllers. Existing
lookup embeddings and routing remain the defaults.

`bert/expert_linear`, `bert/expert_linear_adaptive`, `gpt/expert_linear`, and
`gpt/expert_linear_adaptive` accept `attention_soft_moe_config: SoftMoEConfig`.
In BERT, set it independently of the feed-forward `soft_moe_config`; either
or both can be enabled, with separate scoring stacks and expert parameters.
GPT retains its existing feed-forward routing. For attention in either family:

```python
configuration = package.build_configuration(
    config_overrides={
        "attention_soft_moe_config": attention_routing_config,
        "expert_attention_use_kv_expert_models_flag": False,
    }
)
```

Soft attention pools tokens into query-expert slots, attends from those slots
to shared token keys and values, applies the corresponding output experts,
and reconstructs tokens with the combine weights from the original query
router. Routing is computed once per attention call. Expert count, slots per
expert, and attention head count are independent dimensions; `num_experts`
in the attention `SoftMoEConfig` overrides the attention expert count only.

GPT uses causal prefix slots: the slots for token position `t` pool only
positions up to `t`, and their queries can attend only to those same positions.
Stable cumulative log normalizers produce the current token's slot-update
weights; the previous prefix supplies the remaining weight. Scoring and shared
key/value projections still run once per attention call. The output experts
process each prefix's slots before its token is reconstructed. BERT retains
its bidirectional whole-sequence slots.

The causal path evaluates `sequence_length * slots_per_expert` queries per
expert, with attention weights proportional to
`batch_size * experts * heads * slots_per_expert * sequence_length**2`.
It uses more compute and memory than bidirectional slot attention. Prefixes
are rebuilt on each forward pass; static key/value caches are unsupported.

Both modes support Boolean or additive `0/-inf`
padding masks, and the existing learned key/value bias and zero-attention
source extensions. Padding is excluded from slot pooling and key/value
attention, and padded query outputs are zero. It requires explicit input
layout and pointwise projection stacks. Expert key/value projections,
cross-attention, static key/value inputs, token-pair attention masks,
relative position embeddings for slots, shared adaptive grouping, context
controllers, and returning attention weights are rejected explicitly.
Absolute input positions, package-supported hierarchical embeddings, and
whole-stack reasoning growth can still be used.

`contextual_embedding_config` accepts the same `ByteContextualEmbeddingConfig`
used by GPT, including its byte experts, prefix kernel, contextual experts, and
residual scale. Contextual and hierarchical embeddings are mutually exclusive.
Contextual embeddings supply sentence positions internally; BERT still adds
segment embeddings and keeps its bidirectional encoder. This mode also requires
an untied MLM head and vocabulary binding.

Both adaptive BERT packages share GPT's recurrent reasoning schedule controls:

```python
reasoning_options = {
    "recurrent_flag": True,
    "recurrent_initial_iterations": 2,
    "recurrent_max_steps": 10,
    "recurrent_iteration_increment": 1,
    "recurrent_forward_calls_before_iteration_increment": 20_000,
    "recurrent_smooth_iteration_growth_flag": True,
    "recurrent_no_gradient_transition_count": 0,
}
```

Pass these values through `config_overrides` to start with two iterations and
smoothly grow toward ten, with a 20,000-forward-call growth interval. The counter
tracks successful recurrent-owner forward calls and is saved in checkpoints.
Explicit `recurrent_no_gradient_transition_count=0` keeps gradients through all
iterations; `recurrent_gradient_transition_count` can instead limit the gradient
suffix. Whole-encoder recurrence can be combined with hierarchical embeddings
and Soft MoE feed-forward layers.

`bert/expert_linear_adaptive` also accepts `recurrent_min_steps` and
`recurrent_ponder_cost_weight`, matching its GPT counterpart. Nondefault values
require both `recurrent_flag=True` and `recurrent_stack_halting_flag=True`.
Minimum steps delay early halting; the ponder-cost weight scales the halting
auxiliary loss. These controls are distinct from the scheduled iteration budget.

Each training run writes a Lightning/TensorBoard run directory under:

```text
logs/quickstart/linears/linear/BASELINE/Mnist/<parameter-id>_<timestamp>/version_*/
```

That `version_*` folder contains TensorBoard event files such as
`events.out.tfevents.*`, Lightning metadata such as `hparams.yaml`, and
Emperor's `result.json` with the final parameters and metrics. Checkpoint files
appear there only when checkpointing is enabled.

### Temporary attention memory

All twelve BERT/GPT/ViT packages (`linear`, `linear_adaptive`, `expert_linear`, and
`expert_linear_adaptive` in each family) can append temporary memory to attention.
The feature is disabled by default and applies to the backbone; GPT's hierarchical
byte components remain independently configured.

Alongside the existing Q/K/V projections, attention evaluates
`M = memory_module(X)` once and concatenates the same result to both K and V along
the source sequence dimension. Q is unchanged. A single softmax distributes
weight over ordinary tokens and memory, so the weighted output retains the
original query length. There is one outer memory position per source token.
`AttentionDynamicMemoryConfig.num_memory_slots` controls its internal computation,
not the number of appended positions: its public forward result is used intact.

The package options are:

| Option | Default |
| --- | --- |
| `ATTN_SOURCE_MEMORY_FLAG` | `False` |
| `ATTN_SOURCE_MEMORY_OPTION` | `GatedResidualDynamicMemoryConfig` |
| `ATTN_SOURCE_MEMORY_NUM_MEMORY_SLOTS` | `4` |
| `ATTN_SOURCE_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE` | `None` |
| `ATTN_SOURCE_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS` | `None` |

The generator uses the package's `SUBMODULE_STACK_*` controller-stack settings.
Existing `ATTN_MEMORY_*` options still configure memory inside projection layers;
the two features are independent and can coexist. For example, build BERT, GPT, and ViT
configurations with attention-style source memory without starting training:

```python
from emperor.memory import AttentionDynamicMemoryConfig
from models.catalog import model_package

overrides = {
    "attn_source_memory_flag": True,
    "attn_source_memory_option": AttentionDynamicMemoryConfig,
    "attn_source_memory_num_memory_slots": 3,
}
bert_configuration = model_package("bert/linear").build_configuration(
    config_overrides=overrides,
)
gpt_configuration = model_package("gpt/linear").build_configuration(
    config_overrides=overrides,
)
vit_configuration = model_package("vit/linear").build_configuration(
    config_overrides=overrides,
)
```

The low-level attention `source_memory_config` accepts any of the four concrete
memory configs: `GatedResidualDynamicMemoryConfig`, `WeightedDynamicMemoryConfig`,
`ElementWiseWeightedDynamicMemoryConfig`, or `AttentionDynamicMemoryConfig`.
The package variant selector accepts those same types. Both test-time-training
settings must be supplied together to enable adaptation. Attention flattens the
input into independent token rows, and adaptation uses temporary parameter copies
per token. Generated memory and adapted copies are not retained between calls;
the memory module's ordinary learned parameters are saved in checkpoints.

For `S` source tokens, ordinary attention masks with trailing shape `[Q, S]` and
source padding masks are replicated onto the matching memory positions. An
explicit `[Q, 2S]` attention mask can instead provide a token block followed by a
memory block, including separate finite biases. The memory block cannot reopen
an original token's hard exclusion, padding exclusion, or causal restriction.
In GPT, memory derived from position `j` is visible only to queries at positions
`i >= j`. Soft MoE attention retains its rejection of token-pair attention masks.

RoPE memory keys and relative positional logits reuse the corresponding source
token positions; appending memory does not assign later time positions. Soft MoE
retains its existing restriction on relative positional embeddings. With source
memory enabled, static K/V inputs, cross-attention, incompatible input/key/value
widths, and grouped memory generators are rejected explicitly.

The source length grows from `S` to `2S` before any learned-bias or zero-attention
positions. This roughly doubles the attention-score and weighted-value work and
adds memory-generation work, while leaving the output sequence length unchanged.

### Temporary memory in MLP-Mixer

`MixerAttentionConfig.source_memory_config` is optional and defaults to `None`.
With no memory configuration, no memory module is constructed and the original
token mixer is used. Supplying any of the four concrete memory configurations
enables one temporary memory token per input token.

For input `[B, T, D]`, memory is generated from independent `[B*T, D]` token rows
and reshaped to `[B, T, D]`. Original and memory tokens are concatenated to
`[B, 2T, D]`, transposed and flattened to `[B*D, 2T]`, and passed through a mixer
with `2T` input and output positions. After restoring `[B, 2T, D]`, only the first
`T` output positions are returned. Slicing those positions preserves gradients
from the retained outputs into the memory generator. The public output remains
`[B, T, D]`; sequence-first input is supported too.

All four MLP-Mixer packages expose these options:

| Option | Default |
| --- | --- |
| `TOKEN_MIXER_SOURCE_MEMORY_FLAG` | `False` |
| `TOKEN_MIXER_SOURCE_MEMORY_OPTION` | `GatedResidualDynamicMemoryConfig` |
| `TOKEN_MIXER_SOURCE_MEMORY_NUM_MEMORY_SLOTS` | `4` |
| `TOKEN_MIXER_SOURCE_MEMORY_TEST_TIME_TRAINING_LEARNING_RATE` | `None` |
| `TOKEN_MIXER_SOURCE_MEMORY_TEST_TIME_TRAINING_NUM_INNER_STEPS` | `None` |

For example:

```python
from emperor.memory import ElementWiseWeightedDynamicMemoryConfig
from models.catalog import model_package

configuration = model_package("mlp_mixer/linear").build_configuration(
    config_overrides={
        "token_mixer_source_memory_flag": True,
        "token_mixer_source_memory_option": ElementWiseWeightedDynamicMemoryConfig,
    },
)
```

The source-memory generator uses `SUBMODULE_STACK_*` settings. Existing
`TOKEN_MIXER_MEMORY_*` options continue to configure memory inside the mixing
layers; their dimensional constraints apply to the expanded `2T` mixer.
Attention-style memory slots remain internal and do not change the number of
appended tokens. Optional test-time adaptation uses temporary parameter copies
per token, with both adaptation settings required together.

Generated memory is not cached between calls. Enabled checkpoints include the
memory parameters and larger mixing layers, and must be loaded with the same
configuration. Disabled models retain their existing parameter layout. The
mixer's existing fixed-length, self-processing, and unsupported-mask restrictions
still apply.

### Memory and expert options across model families

GPT, BERT, ViT, and MLP-Mixer each provide all four package variants. The shared
memory and expert choices are available through their public Runtime Defaults
and configuration schema:

| Feature | GPT / BERT / ViT | MLP-Mixer |
| --- | --- | --- |
| Temporary memory, all four variants | `ATTN_SOURCE_MEMORY_*` | `TOKEN_MIXER_SOURCE_MEMORY_*` |
| Layer memory, all four variants | `MEMORY_*`, `ATTN_MEMORY_*`, `FF_MEMORY_*` | `MEMORY_*`, `TOKEN_MIXER_MEMORY_*`, `CHANNEL_MIXER_MEMORY_*` |
| Sparse experts | `expert_linear` and `expert_linear_adaptive`: attention and feed-forward experts | Same expert variants: token and channel mixing experts |
| Soft feed-forward experts, all four variants | `SOFT_MOE_CONFIG` | `SOFT_MOE_CONFIG` for channel mixing |
| Soft attention / token mixing | `ATTENTION_SOFT_MOE_CONFIG` in both expert variants | `TOKEN_MIXER_SOFT_MOE_CONFIG` in all four variants |
| Memory inside sparse experts | `EXPERT_MEMORY_*` | `EXPERT_MEMORY_*` |
| Memory inside adaptive expert routers | `ROUTER_MEMORY_*` in `expert_linear_adaptive` | Uses the existing plain router architecture |

Memory flags default to `False`. Omitting the low-level memory configuration
constructs no memory module. All four concrete memory variants are selectable in
every existing memory location. Each legacy memory prefix now has a
`NUM_MEMORY_SLOTS` setting, defaulting to `4`: for example,
`ATTN_MEMORY_NUM_MEMORY_SLOTS`, `EXPERT_MEMORY_NUM_MEMORY_SLOTS`, and
`CHANNEL_MIXER_MEMORY_NUM_MEMORY_SLOTS`. Only `AttentionDynamicMemoryConfig` uses
this setting. The other memory variants ignore it. MLP-Mixer recurrent memory
inherits the main `MEMORY_NUM_MEMORY_SLOTS` setting.

Each soft expert option takes a `SoftMoEConfig` instance and defaults to `None`,
preserving the package's existing dense or sparse behavior. Attention and
feed-forward routing can be selected independently. GPT feed-forward soft experts
pool a separate set of slots for each causal prefix; future tokens never enter an
earlier prefix. BERT and ViT use bidirectional pooling. The scoring stack and
expert/slot counts retain the same `SoftMoEConfig` interface in all families.

For soft token mixing, MLP-Mixer preserves `[B, D, T]` (or `[B, D, 2T]` with
temporary memory) rather than flattening examples together. Routing pools channel
vectors within each example, and each expert mixes the token coordinates.
The result is restored to the original layout and memory output positions are
discarded. Soft channel mixing instead routes `[B, T, D]`. Both options can be
enabled together with source memory, and gradients reach the memory generator.

These are shared computational options, not identical input architectures.
Text embedding options remain specific to GPT/BERT, image patch options to
ViT/Mixer, and Mixer retains its own token/channel and adaptive-generator settings.
ViT's adaptive expert variant also exposes the shared recurrent minimum-step,
ponder-cost, and residual controls. The public option and execution matrix is
tested in `tests/model_packages/test_memory_expert_parity.py`.

The four encoder/decoder `transformer/*` packages also accept
`ATTN_SOURCE_MEMORY_*`, with optional `ENCODER_ATTN_*` and
`DECODER_SELF_ATTN_*` overrides. The global source-memory setting applies to both
self-attention paths. Cross-attention source memory is explicitly rejected;
ordinary cross-attention continues to read the encoder output. Their attention
and feed-forward layer-memory options also expose `NUM_MEMORY_SLOTS`.

Existing low-level compatibility checks still apply: soft attention uses shared
K/V projections; soft expert stacks must satisfy their pointwise stack contract;
inner recurrent soft expert stacks and arbitrary pairwise attention masks are
unsupported. Outer recurrent Transformer blocks can use soft experts and source
memory. These restrictions produce explicit errors instead of silently dropping
an option. Temporary source memory is independent of memory inside a projection
or expert stack, so enabling source memory does not relax those stack constraints.
Shared layer memory requires equal input, hidden, and output widths; source memory
does not impose that restriction on the feed-forward stack.

### Continuing from a checkpoint

Continue a single Run with `--resume-checkpoint`:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --resume-checkpoint logs/quickstart/linears/linear/BASELINE/Mnist/<run>/version_0/checkpoints/last.ckpt \
  --config --num-epochs 30
```

`NUM_EPOCHS` is the total target, not the number of additional epochs. A
checkpoint saved at epoch 25 has completed 26 epochs, so the new target must be
greater than 26. Continuation requires exactly one `--preset`, exactly one
explicitly selected dataset, and no `--presets`, `--all-presets`, grid search,
or random search.

Only continue from a trusted local Lightning checkpoint. Use the same Model
Package, preset, dataset, compatible Runtime Defaults, and code-compatible
model structure that created it. Emperor copies the opened source into a
private immutable snapshot, binds its SHA-256 digest, applies finite archive,
payload, tensor, and storage admission limits, and strictly loads the selected
model before Lightning restores optimizer, scheduler, precision, loop, and
compatible callback state. Lightning receives that snapshot rather than the
source path.

On Linux, checkpoint decoding also runs under a bounded private decoder process.
This is defense in depth, not a complete hostile-checkpoint sandbox. A service
that accepts checkpoints from untrusted parties must additionally contain the
entire Training Job using deployment-owned process, memory, filesystem, and
cancellation controls. Other platforms retain the trusted-local path unless a
caller explicitly requires isolated decoding through `CheckpointAdmissionPolicy`.

When checkpointing is requested, `last.ckpt` records the most recently
completed epoch while the other retained checkpoint is the best monitored
epoch; these can be different. Continuation writes a new Run Artifact and
records the source filename, epoch, global step, and admitted SHA-256 digest as
lineage in progress events and `result.json`. It never appends to or modifies
the source Run Artifact or checkpoint, and the digest lets the resumed bytes be
verified independently.

Important scaling rules:

- Without `--datasets`, a command trains once for every dataset listed by the
  model's `dataset_options.py`.
- Without `--config --num-epochs 1`, a command uses the model's configured epoch
  count.
- `--config` marks the following flags as model-config overrides.
- Runs multiply by selected presets, selected datasets, and search samples.

For example:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --presets baseline gating \
  --datasets mnist cifar10 \
  --random-search 10 \
  --logdir comparison \
  --config --num-epochs 1
```

That creates `2 x 2 x 10` planned training runs.

Before choosing overrides, print every overridable field and its default value
for the selected model:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-config
```

Common `config` override examples:

```bash
# Smaller/faster hidden stack.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --hidden-dim 64 --stack-num-layers 2 --stack-dropout-probability 0.0

# Training hyperparameters.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --batch-size 64 --learning-rate 0.0003 --trainer-gradient-clip-val 0.5
```

## Running Experiments

Use the canonical `mise` task from the repository root:

```bash
mise run experiment -- \
  --model-type <type> \
  --model <name> \
  [options]
```

Run with no arguments to print the full flag list:

```bash
mise run experiment --
```

List available model types:

```bash
mise run experiment -- --list-model-types
```

List models within a type:

```bash
mise run experiment -- \
  --model-type linears \
  --list-models
```

`--list-models` without `--model-type` still prints the full catalog.

List a model's presets:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-presets
```

List a model's monitor callbacks:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-monitors
```

Run one preset:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline
```

Run selected presets sequentially:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --presets baseline gating memory \
  --datasets mnist \
  --config --num-epochs 1
```

Run every preset for a model:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --all-presets
```

Print a model structure instead of training:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --print-model
```

Run one deterministic synthetic batch and annotate the same tree with every
executed PyTorch module's tensor input and output shapes:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --print-model-shapes
```

For the most detailed trace, also print an executed Python method tree with the
shape of each tensor argument, local variable, and return value, including source
line numbers:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --print-model-tensor-shapes
```

Shape inspection uses a synthetic batch of size 1 in evaluation mode with
gradients disabled. It does not download the selected dataset. Tensor-variable
tracing covers executed Python under `models` and `emperor`; native PyTorch kernel
temporaries are represented by their surrounding module inputs and outputs.

Shape tracing is a trusted, sequential local diagnostic. It executes Model
Package code synchronously in the caller process without a deadline or memory
isolation, temporarily changes the process CPU Torch RNG, and does not isolate
CUDA, Python, NumPy, or arbitrary package side effects. Do not overlap it with
other Torch RNG work. A hosted or untrusted caller must provide process-level
resource and cancellation containment; Workbench does not currently expose
shape tracing.

Monitor callbacks apply to training runs, not model inspection flags.

Run one preset with selected monitors:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --monitors linear halting
```

Use a custom log folder:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --logdir comparison_run
```

Use the same `--logdir` across runs to group TensorBoard logs for comparison.

### Datasets

Each Model Package groups its Dataset Metadata by Experiment Task in
`dataset_options.py`. For `linears/linear`, the image-classification Dataset
Metadata is:

```python
DATASET_OPTIONS_BY_TASK = {
    ExperimentTask.IMAGE_CLASSIFICATION: [
        Mnist,
        FashionMNIST,
        Cifar10,
        Cifar100,
    ],
}
```

BERT-pretraining Dataset Metadata uses the task-specific
`PennTreebankBertPretraining` and `WikiText2BertPretraining` adapters. Their raw
text source Interface accepts both TorchText's legacy `.splits(...)` datasets
and its later `root=..., split=...` factories, while always yielding one text
unit per source line for next-sentence construction. The verified constraints
currently select TorchText 0.6.0.

Causal language modeling exposes `PennTreebank`, `WikiText2`, and `WikiText103`
as download-backed DataModules, with `OpenWebText` available as the much larger
option. WikiText uses the maintained Salesforce datasets on Hugging Face;
OpenWebText uses `Skylion007/openwebtext`. The adapters cache the selected corpus
under their configured root.

Contract tests use deterministic offline source fixtures and do not download
corpora. They discover every declared Dataset Metadata class and verify its task
compatibility, one-batch shape, dtype, collation semantics, and seed ownership
through the public DataModule Interface.

Real downloads remain explicit integration work performed by `prepare_data()`.
Run that network-dependent suite only when the required sources are available:

```bash
EMPEROR_RUN_DATASET_DOWNLOAD_TESTS=1 \
  PYTHONSAFEPATH=1 PYTHONPATH=src:tests \
  python -P -m unittest integration.datasets.test_dataset_downloads
```

OpenWebText is excluded from that integration run unless the additional
`EMPEROR_RUN_LARGE_DATASET_DOWNLOAD_TESTS=1` opt-in is set.

List available datasets for a model:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-datasets
```

Restrict a run to one or more datasets:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist fashion-mnist
```

### Config Overrides

Use `--config` to override model config values without editing `config.py`:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --config --num-epochs 30 --callback-early-stopping-patience 0
```

More examples:

```bash
# Change optimization and batch size.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --batch-size 64 --learning-rate 0.0003

# Make the hidden stack smaller for fast iteration.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --hidden-dim 64 --stack-num-layers 2

# Try a different activation and dropout value.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --stack-activation RELU --stack-dropout-probability 0.1

# Override controller settings when the selected preset enables that controller.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset recurrent-halting \
  --datasets mnist \
  --config --recurrent-max-steps 6 --recurrent-halting-threshold 0.95
```

List overridable fields:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-config
```

### Search

Run a grid search:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --grid-search
```

Run a random search:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --random-search 10
```

Restrict a sweep to selected axes:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --grid-search \
  --search-keys HIDDEN_DIM STACK_NUM_LAYERS
```

Supply command-line search values:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --grid-search \
  --search-set hidden_dim=64,128
```

`--grid-search` and `--random-search` are mutually exclusive and can be combined
with `--preset`, `--presets`, or `--all-presets`.

## Workbench

The Workbench is the local workbench for model experiments. Use it when you want to
pick a model, preset, and dataset visually; inspect what a preset builds before
training; adjust config overrides without assembling a long terminal command;
start or monitor local training jobs; and review historical runs after the
experiment finishes.

It is meant for the human loop around training: understanding the model shape,
checking planned runs, watching monitor signals while a job is active, and
reviewing logs once there are completed results. The terminal CLI remains the
simpler path for scripted or repeatable runs.

The Workbench should be treated as a vibecoded prototype: it is useful for local
research workflows today, but its implementation is expected to be refactored
as the experiment workflow and architecture settle.

See [`apps/workbench/README.md`](apps/workbench/README.md) for the focused
Workbench guide.

Hosted or non-local deployments should configure explicit CORS origins in the
Workbench backend environment:

```bash
export WORKBENCH_API_CORS_ORIGINS='["https://workbench.example.com"]'
```

Hosted frontend builds should also set `NEXT_PUBLIC_WORKBENCH_API_URL` and, when
multiple API origins are allowed, `NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS` so
browser requests and bearer tokens cannot be redirected to arbitrary origins.

Bearer auth can be enabled with:

```bash
export WORKBENCH_API_AUTH_MODE=bearer
export WORKBENCH_API_TOKEN=<token>
```

Example hosted pairing:

```bash
# Frontend build env
export NEXT_PUBLIC_WORKBENCH_API_URL=https://api.example.com
export NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS='["https://api.example.com"]'

# Backend runtime env
export WORKBENCH_API_CORS_ORIGINS='["https://workbench.example.com"]'
export WORKBENCH_API_AUTH_MODE=bearer
export WORKBENCH_API_TOKEN='<replace-with-a-secret-token>'
```

The Workbench backend is local-file-backed. Its in-process locks protect shared
caches inside one backend worker; they are not cross-worker invalidation. Prefer
one backend worker for hosted Workbench deployments unless you add a shared
cache/invalidation design.

## Results and Logs

Training metrics are logged under `logs/`. Launch TensorBoard with:

```bash
tensorboard --logdir logs/
```

Run the archive task from the project directory to archive the contents of
`./logs` directly from the filesystem:

```text
mise run logs:archive
mise run logs:archive -- my_experiment
mise run logs:archive -- logs/my_experiment
```

When a specific experiment folder is selected, the archive keeps that experiment
prefix. Archive members are rooted directly at experiment paths, so importing
restores them beneath `logs/` without a top-level `logs/` wrapper.

By default, the archive is written in the current directory. Pass an explicit
second argument to choose the zip path:

```text
mise run logs:archive -- logs emperor_logs.zip
mise run logs:archive -- logs path/to/emperor_logs.zip
```

To restore that archive into another local project, start the Workbench backend,
open the Workbench, choose **Import Logs** in the top navigation, and select the
produced `.zip` file. Backend defaults keep log imports disabled in every
authentication mode; enable them explicitly with
`WORKBENCH_API_ALLOW_LOG_IMPORTS=true`. The normal `mise run dev` launcher sets
that opt-in for its loopback development backend. The import extracts into that
project's server-side `logs/` directory and overwrites files that already exist
at the same archive paths. Enabled imports default to a 512 MiB compressed limit
and a 2 GiB extracted limit, at most 32,000 archive members, a 4 MiB cumulative
UTF-8 member-path budget, and one active archive import at a time. Override them
with
`WORKBENCH_API_MAX_UPLOAD_SIZE=<bytes>` or
`WORKBENCH_API_MAX_LOG_ARCHIVE_EXTRACTED_SIZE=<bytes>`,
`WORKBENCH_API_MAX_LOG_ARCHIVE_MEMBER_COUNT=<count>`,
`WORKBENCH_API_MAX_LOG_ARCHIVE_PATH_BYTES=<bytes>`, or
`WORKBENCH_API_LOG_ARCHIVE_UPLOAD_CONCURRENCY=<count>`. Size limits always have
finite defaults and cannot be disabled with null configuration values.

## Test and Quality Commands

Use the canonical task from the repository root for the normal unit-test workflow.
It discovers the external `tests/` tree with fail-fast enabled. The script adds
only that test tree to `PYTHONPATH`; Emperor and its first-party Model Packages
must come from the active environment's editable or regular installation. Run
`python -m pip install --no-deps -e .` first when using an environment that was
not created by `mise run setup`.

```text
mise run test
```

Target a specific unit-test module, class, or test method when iterating on a
failure:

```text
mise run test -- layer
mise run test -- layer TestLayer
mise run test -- layer TestLayer test_forward_output_shape
```

The task maps the first argument to `tests/unit/test_<name>.py` and then passes
any class or method name through to `python -P -m unittest -f`.

Verify the protected Emperor interfaces, private-import boundaries, dependency
direction, and zero external retired-interface import ledger with:

```text
mise run test:emperor-interfaces
```

Attention has focused correctness, complete statement-and-branch coverage, and
mutation-strength gates:

```text
mise run test:attention
mise run test:attention-coverage
mise run test:attention-mutation
```

The family runner reports only the Attention modules declared in the test
manifest and requires 100% statement and branch coverage. Mutation scope comes
from the same manifest and the task fails when any mutant remains unreviewed.

Verify the wheel/source-distribution manifests and compare clean editable and
regular installs from outside the checkout with:

```bash
python -P tools/verify_distribution.py
```

The verification uses the dependencies already available in the active
environment and never downloads or upgrades them.

Run the exact locked strict type checker with:

```bash
mise run test:model-runtime-types
```

The strict Pyright include list covers stable capability areas and may only
grow. `mise run setup --profile cpu` provisions Pyright 1.1.411 from the root
Node lockfile. GitHub workflow enforcement is currently deferred, so run this
task locally before committing Runtime changes.

Capture the production Workbench browser and long-session baseline after a
frontend build with:

```bash
cd apps/workbench/web
npm run build
npm run performance:budget
npm run performance:browser
```

The browser harness uses temporary backend data and a fake process only at the
Training Job runner boundary. It measures hydration, main-thread and React
work, repeated workspace/graph/chart cycles, API timings, heap growth, completed
Training and log-import flows, WebGL frames and disposal, and the existing
bundle budgets. Machine-sensitive results are informational; stable workflow,
disposal, error, and bundle checks fail the harness.

For Workbench-specific changes, run the relevant backend or frontend checks from
the package being changed after the unit suite is green.

The live backend/frontend contract E2E uses two loopback FastAPI servers,
temporary filesystem roots, and fake Training Jobs. It requires the verified
Python environment and existing frontend dependencies, but downloads nothing:

```bash
cd apps/workbench/web
npm run test:contract:e2e
```

## License

Except where otherwise noted, the original source code, documentation, and
assets in this repository are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
See [LICENSE](LICENSE) for the complete terms and [NOTICE](NOTICE) for scope,
ownership, and third-party exclusions.
