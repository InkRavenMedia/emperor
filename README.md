# Emperor

**A PyTorch research framework for small, adaptive neural networks.**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](mise.toml)
[![PyTorch 2.12](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white)](#environment-setup)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-4C566A)](LICENSE)

**Current package version:** `0.1.0`

Emperor is a personal machine learning research project for building experimental
models from reusable, inspectable components. Use the CLI to inspect
architectures, train models, compare variants, and review results.

The focus is repeatable experimentation: build small models, measure what
changes, and use the evidence to decide which ideas to explore next.

[Research mission](#research-mission) ·
[Example model](#example-model-adaptive-bert-with-soft-moe-and-attention-memory) ·
[Installation](#environment-setup) · [Quick start](#quick-start) ·
[Experiment CLI](#running-experiments) ·
[Results and logs](#results-and-logs)

## About the author

Emperor is my personal machine learning research project. You can also find me
at [InkRaven](https://inkraven.de/).

## Research mission

Emperor investigates how far architectural improvements can take small neural
networks. The long-term goal is to develop flexible models with capabilities
comparable to frontier AI systems, while keeping them practical to run on a
single GPU.

### The challenge

Increasing model size brings greater demands on memory and computation. This
project explores ways to improve capability through how a model processes
information, uses its parameters, and adapts to new inputs. The challenge is to
find improvements whose benefits hold up when their resource costs are taken
into account.

### How Emperor supports the research

The repository provides a shared foundation for testing those ideas. Reusable
PyTorch components and configurable model packages make it possible to build
variants and compare them with reference models. The CLI supports model
inspection, training, and reviewing results, so architectural choices can
be examined through repeatable experiments.

### Research directions

The work explores latent-space reasoning, adaptive computation, memory,
test-time training, mixture of experts, data-dependent parameters, and dynamic
neuron clusters. These ideas are investigated individually and in combination
to understand what they contribute to small, flexible models.

Progress toward the goal depends on evidence: what a change improves, what it
costs, and where it falls short. Emperor provides the tools to carry out that
investigation; developing the model remains the long-term research objective.

## Example model: adaptive BERT with Soft MoE and attention memory

Emperor can combine these research ideas in a single model. One example is a
small BERT pretraining configuration built from the
[`bert/expert_linear_adaptive`](src/models/bert/expert_linear_adaptive/) package.

The adaptive core contains **136 adaptive linear layers**: 133 use 32 × 32 main
weight matrices, and the three Soft MoE router output layers use 32 × 160
matrices to score 16 experts with 10 slots each. Embeddings, prediction heads,
and parts of the byte encoder use other matrix shapes. The complete
configuration has **5,023,983 trainable parameters** (approximately 5.02 million).

### Research foundations

This configuration includes:

- **BERT pretraining:** a bidirectional encoder trained with masked language
  modeling and next-sentence prediction, following
  [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805).
- **Data-dependent parameters:** adaptive linear layers generate weights, biases,
  and diagonal coefficients from their inputs. This applies to the experts,
  routers, and key/value projections, drawing on
  the parameter-generation principle of
  [HyperNetworks](https://arxiv.org/pdf/1609.09106).
- **Grouped parameter generation:** token-level adaptive layers sum groups of
  five tokens to generate 20 parameter sets for each 100-token sequence. Each
  set is shared by its five tokens, while the output retains all 100 positions.
  The same grouping setting applies inside experts, where their ten Soft MoE
  slots form two groups of five per expert and sequence.
- **Soft mixture of experts:** both attention and feedforward computation use
  learned soft assignments to pool token information into expert slots and
  combine expert outputs back into token representations. This applies the
  slot-routing idea from
  [From Sparse to Soft Mixtures of Experts](https://arxiv.org/pdf/2308.00951),
  extending it to attention in this project.
- **Latent-space reasoning research:** a shared recurrent transformer block
  repeatedly updates the sequence's hidden representations, following the
  recurrence principle of
  [Universal Transformers](https://arxiv.org/abs/1807.03819).
  [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/pdf/2510.04871v1)
  provides related inspiration for iterative latent reasoning; this example
  selects the general recurrent block rather than the TRM architecture.
- **Adaptive halting:** a learned stick-breaking controller determines when
  individual token positions halt, following
  [Sparse Universal Transformer](https://arxiv.org/pdf/2310.07096). A ponder-cost
  penalty accounts for additional computation during training, with
  [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983)
  providing the foundational adaptive-computation idea.
- **Gradual depth growth:** the recurrent computation starts at two iterations
  and can grow smoothly toward a maximum of ten as training progresses.
- **Attention memory:** the sequence is summed in chunks of ten tokens, and an
  independent learned linear model generates one memory embedding per chunk.
  The 100-token example adds ten memory key/value sources while preserving the
  original output sequence length.
- **Attention residual connections:** learned attention weights combine states
  from earlier recurrent computation steps, applying
  [Attention Residuals](https://arxiv.org/pdf/2603.15031) across recurrent depth.
  Residual routing keys use
  [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).
- **Hierarchical byte embeddings:** a byte-level transformer builds token
  representations from their UTF-8 bytes, with learned byte positions and a
  limit of 64 bytes per token. This adapts the byte-to-token encoder idea from
  [Hierarchical Autoregressive Transformers: Combining Byte- and Word-Level Processing for Robust, Adaptable Language Models](https://arxiv.org/abs/2501.10322).

### Run the example

After [GPU environment setup](#environment-setup), run the following Bash
command from the repository root with the WikiText-2 text files available in
`data/wikitext-2/wikitext-2/`. The flags enable the hierarchical input encoder,
Soft MoE in attention and feedforward layers, and attention memory.

<details>
<summary>Full training command (Bash)</summary>

```bash
mise run experiment -- \
  --model-type bert \
  --model expert_linear_adaptive \
  --experiment-task bert-pretraining \
  --preset baseline \
  --datasets wiki-text2-bert-pretraining \
  --logdir bert-025-soft-moe-params-chunk5-memory-chunk10-wiki-text2 \
  --monitors attention recurrent-layer layer-controller adaptive \
  --config \
  --adaptive-generator-stack-hidden-dim 32 \
  --adaptive-generator-stack-num-layers 2 \
  --attn-num-heads 4 \
  --batch-size 16 \
  --bias-option GeneratorDynamicBiasConfig \
  --bias-option-flag true \
  --callback-checkpoint-flag true \
  --callback-early-stopping-patience 0 \
  --data-num-workers 4 \
  --diagonal-option CombinedDynamicDiagonalConfig \
  --diagonal-option-flag true \
  --expert-attention-use-kv-expert-models-flag false \
  --expert-bias-flag true \
  --ff-stack-hidden-dim 32 \
  --generator-depth 5 \
  --hidden-dim 32 \
  --hierarchical-embedding-flag true \
  --hierarchical-byte-embedding-dim 32 \
  --hierarchical-byte-encoder-num-layers 1 \
  --hierarchical-byte-feed-forward-dim 64 \
  --hierarchical-byte-num-heads 4 \
  --hierarchical-max-token-bytes 64 \
  --learning-rate 0.0001 \
  --mask-option-flag false \
  --monitor-log-every-n-steps 500 \
  --num-epochs 100 \
  --num-experts 16 \
  --recurrent-flag true \
  --recurrent-forward-calls-before-iteration-increment 20000 \
  --recurrent-gradient-transition-count None \
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
  --sampler-switch-loss-weight 0.0 \
  --sampler-zero-centred-loss-weight 0.0 \
  --sequence-length 100 \
  --stack-dropout-probability 0.1 \
  --stack-num-layers 1 \
  --trainer-accelerator gpu \
  --trainer-benchmark false \
  --trainer-deterministic false \
  --trainer-devices 1 \
  --trainer-enable-checkpointing true \
  --trainer-enable-model-summary false \
  --trainer-enable-progress-bar false \
  --trainer-gradient-clip-algorithm norm \
  --trainer-gradient-clip-val 1.0 \
  --trainer-max-time 00:11:00:00 \
  --trainer-num-sanity-val-steps 0 \
  --trainer-precision 32-true \
  --weight-decay-rate 0.0001 \
  --weight-decay-schedule EXPONENTIAL \
  --weight-decay-warmup-batches 5000 \
  --weight-normalization-option SOFT_CLAMP \
  --weight-normalization-position-option BEFORE_OUTER_PRODUCT \
  --weight-option DualModelDynamicWeightConfig \
  --weight-option-flag true \
  --attn-source-memory-flag true \
  --attn-source-memory-chunk-size 10 \
  --attn-source-memory-grouping-method SumGroupingConfig \
  --attn-source-memory-model-option LINEAR \
  --attn-source-memory-model-bias-flag true \
  --mlm-decoder-weight-tying-flag false \
  --contextual-embedding-config None \
  --attention-soft-moe-flag true \
  --soft-moe-flag true \
  --soft-moe-slots-per-expert 10 \
  --soft-moe-scoring-hidden-dim 32 \
  --soft-moe-scoring-num-layers 2 \
  --grouping-scope SEQUENCE \
  --chunk-size 5 \
  --grouping-sequence-length 100 \
  --grouping-input-order BATCH_FIRST \
  --grouping-method SumGroupingConfig \
  --attn-grouping-input-order SEQUENCE_FIRST
```

</details>

## Environment setup

Install Git and [mise](https://mise.jdx.dev/), then clone the repository and set
up the CPU environment:

```text
git clone https://github.com/InkRavenMedia/emperor.git
cd emperor
mise run setup --profile cpu
```

Mise provisions the pinned toolchains and project dependencies. Setup creates
`./torchenv`; run the remaining commands from the repository root. Supported
CPU environments are Ubuntu 24.04 x64, WSL2 Ubuntu 24.04 x64, Windows 11 x64
with PowerShell, and macOS 15+ on Apple Silicon. Intel macOS is not supported
by the pinned PyTorch/Python combination.

For an NVIDIA GPU on Linux x86-64, choose one of these profiles instead:

```text
mise run setup --profile cuda
```

The `cuda` profile uses CUDA 13.0 wheels and requires an R580-or-newer driver.
For CUDA 12.6, including the GTX 1080 Ti, use:

```text
mise run setup --profile cuda-legacy
```

The wheels include the CUDA runtime, so a system CUDA toolkit is not required.
The legacy profile needs a compatible NVIDIA driver; see NVIDIA's
[CUDA compatibility requirements](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Switching profiles recreates the project environment. Select GPU training
explicitly with `--config --trainer-accelerator gpu --trainer-devices 1`.

## Quick start

After setup, inspect a `linears/linear` baseline and run a short training job:

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

# 6. Run a one-epoch training example.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --logdir quickstart \
  --config --num-epochs 1
```

This run builds the `baseline` preset, trains it on `mnist` for one epoch,
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

`NUM_EPOCHS` is the total target, not the number of additional epochs.
Continuation requires one preset and one explicitly selected dataset; it
cannot be combined with grid or random search.

Use a trusted local checkpoint with the same model, preset, dataset, and a
compatible configuration. When checkpointing is enabled, `last.ckpt` contains
the most recently completed epoch; the best monitored checkpoint may differ.
Continuation writes a new run and leaves the source checkpoint unchanged.

## Running experiments

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

Use `--print-model-shapes` to inspect module input/output shapes, or
`--print-model-tensor-shapes` for a detailed method trace. Both use synthetic
input without downloading a dataset.

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

Use `--logdir comparison_run` to group runs under a custom log folder.

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

### Datasets

Available datasets depend on the selected model and task. BERT pretraining
uses `penn-treebank-bert-pretraining` or `wiki-text2-bert-pretraining`; GPT
supports `penn-treebank`, `wiki-text2`, `wiki-text103`, and `open-web-text`.
Datasets may be downloaded on first use and are cached locally.

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

### Config overrides

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

Run a ViT classifier with up to three new latent tokens per block invocation:

```bash
mise run experiment -- \
  --model-type vit \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config \
  --sequence-expansion-flag true \
  --sequence-expansion-max-new-tokens 3 \
  --sequence-expansion-model-option LINEAR \
  --sequence-expansion-model-bias-flag true
```

For MLP-Mixer, use `--model-type mlp_mixer` with the same expansion flags.

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

## Results and logs

Training metrics are logged under `logs/`. Launch TensorBoard with:

```bash
mise exec -- python tools/emperor_dev.py python -- -m tensorboard.main --logdir logs/
```

Each training run writes a Lightning/TensorBoard run directory under:

```text
logs/quickstart/linears/linear/BASELINE/Mnist/<parameter-id>_<timestamp>/version_*/
```

That `version_*` folder contains TensorBoard event files such as
`events.out.tfevents.*`, Lightning metadata such as `hparams.yaml`, and
Emperor's `result.json` with the final parameters and metrics. Checkpoint files
appear there only when checkpointing is enabled.

Archive all logs, a named experiment, or choose an output path:

```text
mise run logs:archive
mise run logs:archive -- my_experiment
mise run logs:archive -- logs path/to/emperor_logs.zip
```

## Project structure

| Path | Purpose |
| --- | --- |
| [`src/emperor/`](src/emperor/) | Neural modules, controller primitives, datasets, training helpers, and monitor callbacks. |
| [`src/model_runtime/`](src/model_runtime/) | Shared model-package, inspection, run, and project-adapter interfaces for the CLI. |
| [`src/models/`](src/models/) | Reference experiment packages that compose the core modules into runnable architectures. |

## License

Except where otherwise noted, the original source code, documentation, and
assets in this repository are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
See [LICENSE](LICENSE) for the complete terms and [NOTICE](NOTICE) for scope,
ownership, and third-party exclusions.
