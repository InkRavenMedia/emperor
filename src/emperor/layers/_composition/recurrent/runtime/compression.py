"""Soft dispatch to a shorter sequence for one recurrent transition."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace

import torch
from torch import Tensor, nn

from emperor._validation import (
    _adaptive_grouping_configs,
    recurrent_slot_layout,
    sequence_is_batch_first,
)
from emperor.config import ConfigBase
from emperor.layers._composition.recurrent.config import RecurrentCompressionConfig
from emperor.layers._state import LayerState
from emperor.linears import LinearLayerConfig
from emperor.nn import Module, SoftSlotRouting


def _blocked(mask: Tensor) -> Tensor:
    if mask.dtype == torch.bool:
        return mask
    if mask.is_floating_point() and ((mask == 0) | mask.isneginf()).all():
        return mask.isneginf()
    raise ValueError(
        "Recurrent compression requires boolean or zero/-inf attention masks."
    )


def block_sequence_layout(block: nn.Module, hidden: Tensor) -> tuple[bool, bool]:
    """Respect the transition's attention layout, including implicit causal masks."""
    layouts = set()
    causal = False
    for module in block.modules():
        cfg = getattr(module, "cfg", None)
        layout = getattr(cfg, "batch_first_flag", None)
        if layout is None and hidden.ndim == 3 and hasattr(cfg, "batch_first_flag"):
            layout = sequence_is_batch_first(
                hidden, None, getattr(cfg, "batch_size", None)
            )
        if layout is not None:
            layouts.add(layout)
        causal |= getattr(cfg, "causal_attention_mask_flag", False) is True
    if len(layouts) > 1:
        raise ValueError(
            "Recurrent compression requires a consistent block sequence layout."
        )
    return next(iter(layouts), True), causal


def _self_mask(state: LayerState) -> Tensor | None:
    return getattr(
        state, "target_attention_mask", getattr(state, "attention_mask", None)
    )


def token_padding_mask(state: LayerState) -> Tensor | None:
    mask = getattr(
        state, "target_key_padding_mask", getattr(state, "key_padding_mask", None)
    )
    padding = _blocked(mask) if mask is not None else None
    for key in ("input_mask", "skip_mask"):
        active = getattr(state, key, None)
        if active is not None:
            inactive = ~active.squeeze(-1)
            padding = inactive if padding is None else padding | inactive
    if padding is not None and padding.shape != state.hidden.shape[:-1]:
        raise ValueError(
            "Recurrent compression padding mask must match batch and token dimensions."
        )
    return padding


def requires_prefixes(state: LayerState, implicit_causal: bool) -> bool:
    implicit_causal = implicit_causal or getattr(state, "causal_prefix_flag", False)
    mask = _self_mask(state)
    if mask is None:
        return implicit_causal
    length = state.hidden.size(-2)
    if mask.ndim not in (2, 3) or mask.shape[-2:] != (length, length):
        raise ValueError(
            "Recurrent self-attention masks must match the token sequence."
        )
    blocked = _blocked(mask)
    if not blocked.any():
        return implicit_causal
    causal = torch.ones_like(blocked, dtype=torch.bool).triu(1)
    if not torch.equal(blocked, causal):
        raise ValueError(
            "Recurrent compression supports unrestricted or causal self-attention masks."
        )
    return True


def prefix_state(state: LayerState, end: int) -> LayerState:
    """Take a batch-first target prefix. The source side of cross-attention stays intact."""
    updates = {"hidden": state.hidden[..., :end, :], "loss": None}
    for name in ("key_padding_mask", "target_key_padding_mask"):
        mask = getattr(state, name, None)
        if mask is not None:
            prefix = mask[..., :end]
            endpoint = prefix[..., -1:]
            updates[name] = (
                prefix | endpoint
                if mask.dtype == torch.bool
                else torch.minimum(prefix, endpoint)
            )
    for name in ("attention_mask", "target_attention_mask"):
        mask = getattr(state, name, None)
        if mask is not None:
            updates[name] = mask[..., :end, :end]
    cross_mask = getattr(state, "cross_attention_mask", None)
    if cross_mask is not None:
        updates["cross_attention_mask"] = cross_mask[..., :end, :]
    for key in ("input_mask", "skip_mask"):
        mask = getattr(state, key, None)
        if mask is not None:
            prefix = mask[..., :end, :]
            updates[key] = prefix & prefix[..., -1:, :]
    return replace(state, **updates)


def run_prefixes(
    state: LayerState, transition: Callable[[LayerState], LayerState]
) -> LayerState:
    outputs = []
    loss = state.loss
    padding = token_padding_mask(state)
    for end in range(1, state.hidden.size(-2) + 1):
        endpoint_padding = (
            None if padding is None else padding[..., end - 1 : end, None]
        )
        if endpoint_padding is not None and endpoint_padding.all():
            endpoint = state.hidden[..., end - 1 : end, :]
        else:
            result = transition(prefix_state(state, end))
            endpoint = result.hidden[..., -1:, :]
            loss = LayerState.sum_losses(loss, result.loss)
        if endpoint_padding is not None:
            endpoint = endpoint.masked_fill(endpoint_padding, 0)
        outputs.append(endpoint)
    return replace(state, hidden=torch.cat(outputs, dim=-2), loss=loss)


class RecurrentTokenCompressor(Module):
    def __init__(
        self,
        config: RecurrentCompressionConfig,
        overrides: RecurrentCompressionConfig | None = None,
    ):
        super().__init__()
        if not isinstance(config, RecurrentCompressionConfig):
            raise TypeError("compression_config must be a RecurrentCompressionConfig.")
        self.cfg = config = self._override_config(config, overrides)
        feature_dim = config.input_dim
        if type(feature_dim) is not int or feature_dim <= 0:
            raise ValueError("Compression input_dim must be a positive feature width.")
        if type(config.max_slots) is not int or config.max_slots <= 0:
            raise ValueError(
                "Recurrent compression max_slots must be a positive integer."
            )
        self.max_slots = config.max_slots
        scorer = config.scorer_config
        if scorer is None:
            scorer = LinearLayerConfig(bias_flag=True)
        if not isinstance(scorer, LinearLayerConfig):
            raise TypeError(
                "Compression scorer_config must be a linear or adaptive linear config."
            )
        for name, expected in (
            ("input_dim", feature_dim),
            ("output_dim", self.max_slots),
        ):
            value = getattr(scorer, name)
            if value is not None and value != expected:
                raise ValueError(
                    f"Compression scorer {name} must be {expected}, got {value}."
                )
        scorer = self.configure_block(scorer, include_row_groups=True)
        self.scorer = scorer.build(
            LinearLayerConfig(input_dim=feature_dim, output_dim=self.max_slots)
        )

    def configure_block(
        self, config: ConfigBase, *, include_row_groups: bool = False
    ) -> ConfigBase:
        if not any(_adaptive_grouping_configs(config, root="compression")):
            return config
        resolved = deepcopy(config)
        for _, augmentation in _adaptive_grouping_configs(
            resolved, root="compressed block"
        ):
            grouping = augmentation.grouping_config
            if include_row_groups or grouping.scope.name == "SEQUENCE":
                augmentation.grouping_config = grouping.for_recurrent_slots()
        return resolved

    def forward(
        self, block: nn.Module, state: LayerState, *, complete_prefix: bool = False
    ) -> LayerState:
        if (
            getattr(state, "probabilities", None) is not None
            or getattr(state, "indices", None) is not None
        ):
            raise ValueError(
                "Compressed blocks must compute expert routing on their slot inputs."
            )
        if state.hidden.ndim < 2:
            raise ValueError(
                "Recurrent compression requires a feature-last token sequence."
            )
        batch_first, implicit_causal = block_sequence_layout(block, state.hidden)
        sequence_first = not batch_first and state.hidden.ndim >= 3
        if sequence_first and state.hidden.ndim > 3:
            raise ValueError("Sequence-first attention expects one batch dimension.")
        hidden = state.hidden.transpose(0, 1) if sequence_first else state.hidden
        canonical = replace(state, hidden=hidden)
        for key in ("input_mask", "skip_mask"):
            mask = getattr(state, key, None)
            if sequence_first and mask is not None:
                setattr(canonical, key, mask.transpose(0, 1))
        if hidden.ndim > 3:
            canonical.hidden = hidden.flatten(0, -3)
            for key in (
                "target_key_padding_mask",
                "key_padding_mask",
                "input_mask",
                "skip_mask",
            ):
                mask = getattr(canonical, key, None)
                if mask is not None:
                    setattr(canonical, key, mask.flatten(0, hidden.ndim - 3))

        def transition(inputs: LayerState) -> LayerState:
            return self._process_sequences(block, inputs, sequence_first)

        if canonical.hidden.size(-2) == 0:
            return state
        if not complete_prefix and requires_prefixes(canonical, implicit_causal):
            result = run_prefixes(canonical, transition)
        else:
            result = transition(canonical)
        restored = result.hidden.reshape_as(hidden)
        if sequence_first:
            restored = restored.transpose(0, 1)
        return replace(state, hidden=restored, loss=result.loss)

    def _process_sequences(
        self, block: nn.Module, state: LayerState, sequence_first: bool
    ) -> LayerState:
        unbatched = state.hidden.ndim == 2
        tokens = state.hidden.unsqueeze(0) if unbatched else state.hidden
        batch_size, token_count, _ = tokens.shape
        padding = token_padding_mask(state)
        if padding is not None and unbatched:
            padding = padding.unsqueeze(0)

        loss = state.loss
        if padding is None and batch_size:
            batch = torch.arange(batch_size, device=tokens.device)
            output, auxiliary = self._run_bucket(
                block, state, tokens, batch, batch_size, sequence_first
            )
            loss = LayerState.sum_losses(loss, auxiliary)
        else:
            # A single block call per valid length keeps padding out of both models.
            valid = (
                ~padding
                if padding is not None
                else tokens.new_ones((batch_size, token_count), dtype=torch.bool)
            )
            counts = valid.sum(-1)
            output = tokens.masked_fill(~valid[..., None], 0)
            for count in counts.unique().tolist():
                if count == 0:
                    continue
                batch = (counts == count).nonzero(as_tuple=True)[0]
                positions = valid[batch].nonzero(as_tuple=True)[1].reshape(-1, count)
                features = tokens[batch[:, None], positions]
                restored, auxiliary = self._run_bucket(
                    block, state, features, batch, batch_size, sequence_first
                )
                output = output.index_put((batch[:, None], positions), restored)
                loss = LayerState.sum_losses(loss, auxiliary)
        return replace(
            state, hidden=output.squeeze(0) if unbatched else output, loss=loss
        )

    def _run_bucket(
        self,
        block: nn.Module,
        state: LayerState,
        features: Tensor,
        batch: Tensor,
        batch_size: int,
        sequence_first: bool,
    ) -> tuple[Tensor, Tensor | None]:
        slot_count = min(features.size(1), self.max_slots)
        scored = self._score(features)
        logits = scored.hidden[..., :slot_count]
        routing = SoftSlotRouting()
        slot_hidden = routing.compress_tokens(features, logits)
        unbatched = state.hidden.ndim == 2
        if unbatched:
            slot_hidden = slot_hidden.squeeze(0)
        elif sequence_first:
            slot_hidden = slot_hidden.transpose(0, 1)
        slot_state = self._slot_state(
            state, slot_hidden, batch, batch_size, slot_count, sequence_first
        )
        with recurrent_slot_layout(slot_count, batch_first=not sequence_first):
            result = block(slot_state)
        if (
            not isinstance(result, LayerState)
            or not isinstance(result.hidden, Tensor)
            or result.hidden.shape != slot_hidden.shape
        ):
            raise ValueError(
                "A compressed recurrent block must preserve its slot sequence shape."
            )
        if (
            result.hidden.dtype != slot_hidden.dtype
            or result.hidden.device != slot_hidden.device
        ):
            raise ValueError(
                "A compressed recurrent block must preserve hidden dtype and device."
            )
        if getattr(result, "sequence_expansions", ()):
            raise ValueError(
                "Sequence expansion inside a compressed recurrent block has no token mapping."
            )
        processed = result.hidden
        if unbatched:
            processed = processed.unsqueeze(0)
        elif sequence_first:
            processed = processed.transpose(0, 1)
        return routing.decompress_slots(processed), LayerState.sum_losses(
            scored.loss, result.loss
        )

    def _score(self, features: Tensor) -> LayerState:
        # Adaptive generators may share context or route rows. Keep examples independent.
        if getattr(self.scorer, "adaptive_cfg", None) is not None:
            with recurrent_slot_layout(features.size(1)):
                rows = [
                    LayerState.run_tensor_module(self.scorer, example)
                    for example in features
                ]
            return LayerState(
                torch.stack([row.hidden for row in rows]),
                LayerState.sum_losses(*(row.loss for row in rows)),
            )
        rows = LayerState.run_tensor_module(self.scorer, features.flatten(0, 1))
        return replace(
            rows, hidden=rows.hidden.reshape(*features.shape[:2], self.max_slots)
        )

    @staticmethod
    def _slot_state(
        state: LayerState,
        hidden: Tensor,
        batch: Tensor,
        batch_size: int,
        slot_count: int,
        sequence_first: bool,
    ) -> LayerState:
        updates = {
            "hidden": hidden,
            "loss": None,
            "residual_state": None,
            "halting_state": None,
        }
        for name in (
            "attention_mask",
            "target_attention_mask",
            "key_padding_mask",
            "target_key_padding_mask",
        ):
            if hasattr(state, name):
                updates[name] = None
        for key in ("input_mask", "skip_mask"):
            if getattr(state, key, None) is not None:
                updates[key] = torch.ones_like(hidden[..., :1], dtype=torch.bool)
        if hasattr(state, "sequence_expansions"):
            updates.update(sequence_expansions=(), _expansion_position_offset=None)
        encoder = getattr(state, "encoder_output", None)
        if encoder is not None and encoder.ndim == 3:
            updates["encoder_output"] = encoder.index_select(
                1 if sequence_first else 0, batch
            )
        padding = getattr(state, "encoder_padding_mask", None)
        if padding is not None and padding.ndim == 2:
            updates["encoder_padding_mask"] = padding.index_select(0, batch)
        cross = getattr(state, "cross_attention_mask", None)
        if cross is not None:
            first = cross[..., :1, :]
            if not torch.equal(cross, first.expand_as(cross)):
                raise ValueError(
                    "Recurrent compression requires shared cross-attention source restrictions across target positions."
                )
            if first.ndim == 3 and first.size(0) != 1:
                first = (
                    first.reshape(batch_size, -1, 1, first.size(-1))
                    .index_select(0, batch)
                    .flatten(0, 1)
                )
            updates["cross_attention_mask"] = first.expand(
                *first.shape[:-2], slot_count, first.size(-1)
            )
        return replace(state, **updates)
