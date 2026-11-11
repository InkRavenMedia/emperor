# Standalone masked-word MLP-Mixer

Select `mlp_mixer/expert_linear_adaptive` for WikiText-103 hierarchical masked-word
pretraining. `bert-pretraining` names the shared masked-word plus next-sentence
prediction task; it does not select or import a BERT model implementation.
The BERT Model Packages are unchanged.

The text path uses this package's Mixer backbone. Sequence-aware adaptive Soft
MoE projects token rows before a dense MLP mixes positions. Channel mixing uses
its own experts. This preserves sequence/chunk grouping on token rows and Soft
MoE slots. It avoids applying token grouping to transposed embedding channels.
The original image classifier and its transposed token experts remain available.

The hierarchical byte encoder and causal byte decoder retain their byte-local
attention, matching BERT-002's input and prediction boundary. Sentence-level
attention is replaced by MLP mixing. This is not an entirely attention-free byte
pipeline. Byte reconstruction, next-sentence prediction, exact-word validation
accuracy, padding and oversized-word handling follow the same task contract.

## BERT-002 feature mapping

The example below maps BERT-002-C01's normalized Runtime Defaults, recorded at
revision `397c8a1f`, to native Mixer options. It is a fresh-run example: the
historical continuation timer is cleared. It is not a checkpoint migration or
execution authorization, and it has not been launched.

| Reference setting | Standalone Mixer setting |
| --- | --- |
| Attention Soft MoE | `token_mixer_soft_moe_flag` |
| Feed-forward Soft MoE | `soft_moe_flag` in the channel branch |
| Attention source memory | `token_mixer_source_memory_*` with chunk size 5 |
| Feed-forward hidden width | `channel_mixer_stack_hidden_dim` |
| Grouped adaptive weights and biases | Common `grouping_*` / `chunk_size`; optional channel-specific overrides |
| Recurrent schedule and Attention Residual | Same `recurrent_*` settings |
| Stick Breaking minimum steps and ponder cost | `recurrent_min_steps`, `recurrent_ponder_cost_weight` |
| MLP over positions | `token_mixing_hidden_dim`, `token_mixing_num_layers` |

Text selection enables hierarchical masked-word input and token-row Soft MoE.
`token_mixer_num_layers` controls the expert-projection stack; the separate
`token_mixing_num_layers` controls the dense MLP over positions. For text,
`channel_mixer_num_layers` is the number of channel mixture layers;
`expert_stack_num_layers` controls depth within each expert. The legacy image
path retains its mirrored-stack interpretation. The example explicitly
matches the reference's normalization, expert depth and halting controller,
since the image Mixer's defaults differ. It uses 16 experts, 10 slots per
expert, 100 positions, width 32, one block, adaptive weight/bias generators,
five-token Sum grouping, chunked source memory and recurrent halting.

A positive `token_mixer_source_memory_chunk_size` selects temporary chunk memory:
one generated row per chunk, including a partial final chunk. Padding does not
contribute. At sequence capacity 100 and chunk size 5, the token MLP mixes 120
positions and returns the original 100. With no chunk size, the existing per-token
Dynamic Memory path remains available; its test-time-training controls cannot be
combined with chunked memory.

Backbone head counts, Q/K/V projection controls and KV caches have no Mixer
counterpart. The text path requires token-row Soft MoE and rejects sequence
expansion; neither is silently substituted. Inputs may be shorter than the
configured capacity, with padding masks, but cannot exceed it. Existing BERT
checkpoints are not loadable as Mixer checkpoints.

## Fresh-run command (not executed)

```bash
mise run experiment -- \
  --model-type mlp_mixer \
  --model expert_linear_adaptive \
  --experiment-task bert-pretraining \
  --preset baseline \
  --datasets wiki-text103-bert-hierarchical \
  --logdir mlp-mixer-bert-002-equivalent \
  --config \
    --adaptive-generator-stack-hidden-dim 32 \
    --adaptive-generator-stack-num-layers 2 \
    --batch-size 16 \
    --bias-option GeneratorDynamicBiasConfig \
    --bias-option-flag true \
    --callback-checkpoint-flag true \
    --callback-early-stopping-patience 0 \
    --chunk-size 5 \
    --data-num-workers 4 \
    --diagonal-option-flag false \
    --expert-bias-flag true \
    --generator-depth 5 \
    --grouping-input-order BATCH_FIRST \
    --grouping-method SumGroupingConfig \
    --grouping-scope SEQUENCE \
    --grouping-sequence-length 100 \
    --hidden-dim 32 \
    --hierarchical-byte-embedding-dim 32 \
    --hierarchical-byte-encoder-num-layers 1 \
    --hierarchical-byte-feed-forward-dim 64 \
    --hierarchical-byte-num-heads 4 \
    --hierarchical-masked-word-flag true \
    --hierarchical-max-token-bytes 64 \
    --learning-rate 0.0001 \
    --mask-option-flag false \
    --monitor-log-every-n-steps 500 \
    --num-epochs 100 \
    --num-experts 16 \
    --recurrent-flag true \
    --recurrent-forward-calls-before-iteration-increment 20000 \
    --recurrent-gradient-transition-count null \
    --recurrent-halting-hidden-state-mode RAW \
    --recurrent-halting-option StickBreakingConfig \
    --recurrent-halting-threshold 0.999 \
    --recurrent-initial-iterations 2 \
    --recurrent-iteration-increment 1 \
    --recurrent-max-steps 10 \
    --recurrent-min-steps 2 \
    --recurrent-no-gradient-transition-count 0 \
    --recurrent-ponder-cost-weight 0.01 \
    --recurrent-residual-block-size 1 \
    --recurrent-residual-connection-option AttentionResidualConfig \
    --recurrent-residual-rms-norm-epsilon 1e-06 \
    --recurrent-smooth-iteration-growth-flag true \
    --recurrent-stack-halting-flag true \
    --run-test-after-fit false \
    --sampler-switch-loss-weight 0 \
    --sampler-zero-centred-loss-weight 0 \
    --sequence-expansion-flag false \
    --sequence-length 100 \
    --soft-moe-flag true \
    --soft-moe-scoring-hidden-dim 32 \
    --soft-moe-scoring-num-layers 2 \
    --soft-moe-slots-per-expert 10 \
    --stack-dropout-probability 0.1 \
    --stack-num-layers 1 \
    --trainer-accelerator cpu \
    --trainer-benchmark false \
    --trainer-deterministic false \
    --trainer-devices 1 \
    --trainer-enable-checkpointing true \
    --trainer-enable-model-summary false \
    --trainer-enable-progress-bar false \
    --trainer-gradient-clip-algorithm norm \
    --trainer-gradient-clip-val 1 \
    --trainer-limit-val-batches 0.02 \
    --trainer-log-every-n-steps 10 \
    --trainer-max-time null \
    --trainer-num-sanity-val-steps 0 \
    --trainer-precision 32-true \
    --trainer-val-check-interval 0.00025 \
    --weight-decay-rate 0.0001 \
    --weight-decay-schedule EXPONENTIAL \
    --weight-decay-warmup-batches 5000 \
    --weight-normalization-learnable-scale-flag false \
    --weight-normalization-option SOFTSIGN \
    --weight-normalization-position-option DISABLED \
    --weight-option BroadcastSumDynamicWeightConfig \
    --weight-option-flag true \
    --token-mixer-soft-moe-flag true \
    --token-mixer-source-memory-chunk-size 5 \
    --token-mixer-source-memory-flag true \
    --token-mixer-source-memory-grouping-method SumGroupingConfig \
    --token-mixer-source-memory-model-config '{"bias_flag":true,"type":"LinearLayerConfig"}' \
    --channel-mixer-stack-hidden-dim 32 \
    --layer-norm-position AFTER \
    --token-mixer-num-layers 1 \
    --token-mixer-stack-hidden-dim 32 \
    --token-mixer-stack-layer-norm-position AFTER \
    --token-mixer-stack-dropout-probability 0.1 \
    --token-mixer-stack-apply-output-postprocessing-flag true \
    --channel-mixer-num-layers 1 \
    --channel-mixer-stack-layer-norm-position DISABLED \
    --channel-mixer-stack-dropout-probability 0.0 \
    --channel-mixer-stack-apply-output-postprocessing-flag false \
    --expert-stack-num-layers 2 \
    --expert-stack-apply-output-postprocessing-flag true \
    --submodule-stack-num-layers 2 \
    --submodule-stack-layer-norm-position AFTER
```

## Verification

`tests/model_packages/test_mlp_mixer_masked_words.py` checks the public package and
CLI, configuration parity outside the replaced backbone operation, shared task
inputs/metrics, byte and NSP losses, expert/router/generator/memory gradients,
one optimizer step, checkpoint round trips and recurrence schedule persistence.
It also checks sample isolation, padding, shorter sequences and absence of a
runtime import of a BERT implementation. `tests/unit/test_mixer_chunked_memory.py`
checks chunk sums, partial chunks, masking and memory gradients. Image package
regression tests cover the retained classifier path. These are local engineering
checks; they provide no training-quality or performance result.
