"""Descriptions and section names displayed by configuration Inspection."""

EXPLICIT_FIELD_DESCRIPTIONS = {
    "RESIDUAL_COEFFICIENT_ACTIVATION": (
        "Activation applied to scalar and model-generated coefficients in weighted "
        "and weighted-blend residuals throughout this model. SIGMOID is recommended "
        "and selected explicitly by the model recipe. DISABLED uses the raw "
        "coefficient; other activations may produce negative or unbounded weights. "
        "This does not change the activation inside the coefficient model."
    ),
    "ATTN_SOURCE_MEMORY_FLAG": (
        "Enables memory embeddings generated from grouped source chunks. "
        "Set a chunk size to control how many source tokens each embedding summarizes."
    ),
    "ATTN_SOURCE_MEMORY_MODEL_OPTION": (
        "Selects the generator that turns each grouped source chunk into a memory "
        "embedding. Choose a linear projection, a linear stack, or a recurrent linear layer."
    ),
    "ATTN_SOURCE_MEMORY_MODEL_BIAS_FLAG": (
        "Sets whether the linear source-memory generator includes a bias term."
    ),
    "ATTN_SOURCE_MEMORY_GROUPING_MODEL_OPTION": (
        "Selects the scorer for attention grouping or the statistics projection for "
        "mean/std grouping. Choose LINEAR for either learned grouping method."
    ),
    "SEQUENCE_EXPANSION_FLAG": (
        "Enables count prediction and latent-token expansion before each transformer "
        "block invocation, including supported recurrent passes. Defaults to false."
    ),
    "SEQUENCE_EXPANSION_CONFIG": (
        "Supplies a complete SequenceExpansionConfig as structured configuration JSON. "
        "Individual expansion settings override the corresponding fields."
    ),
    "SEQUENCE_EXPANSION_MAX_NEW_TOKENS": (
        "Sets the per-invocation maximum. Each sample adds the most probable count "
        "from 1 through this value; it is not an overall sequence-length limit."
    ),
    "SEQUENCE_EXPANSION_MODEL_CONFIG": (
        "Preserves a structured count-predictor configuration from an existing setup. "
        "Use the model option and bias flag to configure a linear predictor."
    ),
    "SEQUENCE_EXPANSION_MODEL_OPTION": (
        "Selects the count predictor. Linear and adaptive predictors can use one layer "
        "or a stack. The block supplies the input width and output count."
    ),
    "SEQUENCE_EXPANSION_MODEL_BIAS_FLAG": (
        "Sets whether the count predictor includes bias. Adaptive predictors require bias."
    ),
    "SEQUENCE_EXPANSION_POSITIONAL_EMBEDDING_CONFIG": (
        "Optionally supplies learned or sinusoidal text positions for generated tokens "
        "as structured JSON. None leaves positions disabled unless the complete "
        "expansion configuration supplies them."
    ),
    "SEQUENCE_EXPANSION_POSITIONAL_EMBEDDING_OPTION": (
        "Selects learned or sinusoidal positions for newly generated tokens. "
        "Choose DISABLED to override inherited positions on a stack."
    ),
    "SEQUENCE_EXPANSION_POSITIONAL_EMBEDDING_NUM_EMBEDDINGS": (
        "Sets the initial positional table capacity for generated tokens."
    ),
    "HIERARCHICAL_EMBEDDING_FLAG": (
        "Enables the hierarchical byte encoder. For BERT, turn off MLM decoder "
        "weight tying to use this embedding."
    ),
    "CONTEXTUAL_EMBEDDING_FLAG": (
        "Enables contextual byte embeddings. For BERT, turn off MLM decoder "
        "weight tying to use this embedding."
    ),
    "BATCH_SIZE": (
        "Sets the number of examples processed in each training batch. Larger "
        "batches can improve throughput but use more memory and change optimizer "
        "dynamics."
    ),
    "NUM_EPOCHS": (
        "Sets the maximum number of full passes over the training dataset. "
        "Training can still stop earlier when trainer limits, early stopping, "
        "or max step settings are active."
    ),
    "LEARNING_RATE": (
        "Sets the optimizer step size for parameter updates. Lower values train "
        "more cautiously; higher values can converge faster but may destabilize "
        "training."
    ),
    "INPUT_DIM": (
        "Defines the flattened input feature dimension expected by the model. "
        "This should match the selected dataset or input projection."
    ),
    "OUTPUT_DIM": (
        "Defines the number of output logits produced by the model. For "
        "classification runs this should match the selected dataset class count."
    ),
    "RECURRENT_GRADIENT_TRANSITION_COUNT": (
        "Sets how many final active recurrent transitions track gradients. Earlier "
        "transitions run without gradients. A positive value with smooth growth "
        "selects the legacy source/target branch handoff."
    ),
    "RECURRENT_NO_GRADIENT_TRANSITION_COUNT": (
        "Sets how many initial recurrent transitions run without gradients. "
        "Explicit zero keeps every executed transition in one autograd chain and, "
        "with smooth growth and no gradient suffix, selects the nested handoff."
    ),
    "RECURRENT_SMOOTH_ITERATION_GROWTH_FLAG": (
        "Controls adjacent-depth interpolation while recurrent depth grows. A "
        "positive gradient suffix uses the legacy branch handoff; explicit zero for "
        "the no-gradient prefix uses the full-gradient nested handoff."
    ),
    "RUN_TEST_AFTER_FIT": (
        "Controls whether the trainer runs the test loop after fitting finishes. "
        "Disable this for quick smoke tests when you only need to verify training "
        "and validation."
    ),
    "TRAINER_ACCELERATOR": (
        "Selects the hardware backend used by PyTorch Lightning, such as CPU, "
        "GPU, or auto-detection. Use this when forcing a run onto specific "
        "hardware."
    ),
    "TRAINER_DEVICES": (
        "Selects how many devices or which device IDs the trainer should use. "
        "Keep this aligned with the chosen accelerator and available hardware."
    ),
    "TRAINER_GRADIENT_CLIP_VAL": (
        "Sets the gradient clipping threshold used before optimizer steps. Use a "
        "positive value to reduce exploding gradients in unstable runs."
    ),
    "TRAINER_GRADIENT_CLIP_ALGORITHM": (
        "Chooses how gradient clipping is applied, such as clipping by norm or "
        "by value. This only matters when the gradient clipping value is active."
    ),
    "TRAINER_ACCUMULATE_GRAD_BATCHES": (
        "Accumulates gradients across this many batches before each optimizer "
        "step. Values above 1 simulate a larger effective batch size but make "
        "max steps consume more batches."
    ),
    "TRAINER_PRECISION": (
        "Controls numeric precision for training, such as full precision or "
        "mixed precision. Changing this can affect memory use, speed, and "
        "numerical stability."
    ),
    "TRAINER_DETERMINISTIC": (
        "Requests deterministic operations from the trainer where possible. "
        "Enable this for reproducibility checks, with the tradeoff that some "
        "operations may run slower."
    ),
    "TRAINER_BENCHMARK": (
        "Allows backend benchmarking to choose faster kernels for fixed input "
        "shapes. Disable this when deterministic behavior is more important "
        "than throughput."
    ),
    "TRAINER_MAX_STEPS": (
        "Stops training after this many optimizer steps. Use a positive value "
        "for smoke tests or bounded experiments; -1 lets training run according "
        "to the epoch limit."
    ),
    "TRAINER_MAX_TIME": (
        "Stops training after a wall-clock time budget accepted by PyTorch "
        "Lightning. Leave it unset when training should be bounded by epochs or "
        "steps instead."
    ),
    "TRAINER_VAL_CHECK_INTERVAL": (
        "Controls how often validation runs during training. Use 1.0 for once "
        "per epoch, a fraction for within-epoch checks, or a batch interval when "
        "supported by the trainer."
    ),
    "TRAINER_LIMIT_TRAIN_BATCHES": (
        "Limits how much training data is used in each epoch. Use a fraction "
        "such as 0.1 for quick smoke tests, or an integer batch count when you "
        "need a fixed small run."
    ),
    "TRAINER_LIMIT_VAL_BATCHES": (
        "Limits how much validation data is evaluated each validation pass. Use "
        "a fraction such as 0.1 for 10% of validation batches during quick "
        "checks."
    ),
    "TRAINER_OVERFIT_BATCHES": (
        "Forces training and validation to reuse a tiny batch subset. Use this "
        "to check whether the model can overfit before running a full training "
        "job."
    ),
    "TRAINER_NUM_SANITY_VAL_STEPS": (
        "Runs this many validation batches before training starts. These sanity "
        "checks catch validation-loop errors before spending time on fitting."
    ),
    "TRAINER_LOG_EVERY_N_STEPS": (
        "Controls how often trainer metrics are logged in training steps. Lower "
        "values give more granular logs but increase logging overhead."
    ),
    "MONITOR_LOG_EVERY_N_STEPS": (
        "Controls how often selected monitor callbacks emit TensorBoard data in "
        "training steps. Lower values make monitor charts denser but add logging "
        "overhead."
    ),
    "TRAINER_ENABLE_PROGRESS_BAR": (
        "Controls whether PyTorch Lightning renders a progress bar. Disable it "
        "for quieter logs in automated or Workbench-launched runs."
    ),
    "TRAINER_ENABLE_CHECKPOINTING": (
        "Controls Lightning's built-in checkpointing behavior. This is separate "
        "from explicit Workbench or callback checkpoint settings."
    ),
    "TRAINER_ENABLE_MODEL_SUMMARY": (
        "Controls whether the trainer prints a model summary before training. "
        "Enable it when inspecting layer structure; disable it to keep logs "
        "compact."
    ),
    "TRAINER_PROFILER": (
        "Selects a PyTorch Lightning profiler to collect runtime performance "
        "details. Leave it unset for normal runs because profiling adds overhead."
    ),
    "DATA_NUM_WORKERS": (
        "Sets the number of worker processes used by supported data modules for "
        "loading batches. Increase it to improve input throughput, or use 0 for "
        "simpler debugging."
    ),
    "CALLBACK_EARLY_STOPPING_PATIENCE": (
        "Sets how many validation checks can pass without improvement before "
        "early stopping halts training. Use 0 to disable the early stopping "
        "callback."
    ),
    "CALLBACK_EARLY_STOPPING_METRIC": (
        "Selects the logged validation metric watched by early stopping and "
        "checkpoint callbacks. The metric name must be emitted during "
        "validation."
    ),
    "CALLBACK_EARLY_STOPPING_MODE": (
        "Sets the improvement direction for early stopping and checkpoint selection. "
        "Use min for a decreasing metric or max for an increasing metric. Leave unset "
        "to use the monitored metric's default direction."
    ),
    "CALLBACK_EARLY_STOPPING_MIN_DELTA": (
        "Sets the minimum metric change that counts as an improvement for early "
        "stopping. Increase it to ignore tiny fluctuations."
    ),
    "CALLBACK_EARLY_STOPPING_STRICT": (
        "Controls whether missing early-stopping metrics raise an error. Keep it "
        "enabled when a missing metric should fail the run loudly."
    ),
    "CALLBACK_EARLY_STOPPING_CHECK_FINITE": (
        "Controls whether early stopping aborts when the monitored metric "
        "becomes NaN or infinite. Keep it enabled to fail unstable runs quickly."
    ),
    "CALLBACK_CHECKPOINT_FLAG": (
        "Controls whether a model checkpoint callback is added for the selected "
        "run. When enabled, checkpoint selection follows the configured early "
        "stopping metric."
    ),
    "STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG": (
        "Controls whether the main layer stack applies activation, dropout, layer "
        "normalization, and residual processing to its final layer."
    ),
    "HALTING_STACK_APPLY_OUTPUT_POSTPROCESSING_FLAG": (
        "Controls whether the dedicated halting stack applies activation, dropout, "
        "layer normalization, and residual processing to its final layer. Only "
        "matters when halting uses an independent stack; None lets the builder "
        "inherit the default stack behavior."
    ),
    "HALTING_STACK_BIAS_FLAG": (
        "Controls whether linear layers in the dedicated halting stack include "
        "bias terms. Only matters when halting uses its own stack; None lets the "
        "builder inherit the default stack behavior."
    ),
}

SECTION_CONTEXT = {
    "Layer Stack Options": "main layer stack",
    "Layer Stack Submodule Options": "shared layer-stack submodules",
    "Adaptive Generator Stack Options": "adaptive-parameter generator submodules",
    "Gate Options": "layer gating controller",
    "Gate Stack Options": "dedicated gate stack",
    "Halting Options": "halting controller",
    "Halting Stack Options": "dedicated halting stack",
    "Memory Options": "dynamic memory controller",
    "Memory Stack Options": "dedicated memory stack",
    "Recurrent Layer Options": "recurrent wrapper",
    "Recurrent Gate Options": "recurrent gate controller",
    "Recurrent Gate Stack Options": "dedicated recurrent gate stack",
    "Recurrent Halting Options": "recurrent halting controller",
    "Recurrent Halting Stack Options": "dedicated recurrent halting stack",
    "Feed-Forward Stack Options": "transformer feed-forward stack",
    "Feed-Forward Gate Options": "feed-forward gate controller",
    "Feed-Forward Gate Stack Options": "dedicated feed-forward gate stack",
    "Feed-Forward Halting Options": "feed-forward halting controller",
    "Feed-Forward Halting Stack Options": ("dedicated feed-forward halting stack"),
    "Feed-Forward Memory Options": "feed-forward dynamic memory controller",
    "Feed-Forward Memory Stack Options": ("dedicated feed-forward memory stack"),
    "Feed-Forward Recurrent Layer Options": "feed-forward recurrent wrapper",
    "Feed-Forward Recurrent Gate Options": ("feed-forward recurrent gate controller"),
    "Feed-Forward Recurrent Gate Stack Options": (
        "dedicated feed-forward recurrent gate stack"
    ),
    "Feed-Forward Recurrent Halting Options": (
        "feed-forward recurrent halting controller"
    ),
    "Feed-Forward Recurrent Halting Stack Options": (
        "dedicated feed-forward recurrent halting stack"
    ),
    "Attention Projection Stack Options": "attention projection stack",
    "Attention Projection Gate Options": "attention projection gate controller",
    "Attention Projection Gate Stack Options": (
        "dedicated attention projection gate stack"
    ),
    "Attention Projection Halting Options": ("attention projection halting controller"),
    "Attention Projection Halting Stack Options": (
        "dedicated attention projection halting stack"
    ),
    "Attention Projection Memory Options": (
        "attention projection dynamic memory controller"
    ),
    "Attention Projection Memory Stack Options": (
        "dedicated attention projection memory stack"
    ),
    "Attention Projection Recurrent Layer Options": (
        "attention projection recurrent wrapper"
    ),
    "Attention Projection Recurrent Gate Options": (
        "attention projection recurrent gate controller"
    ),
    "Attention Projection Recurrent Gate Stack Options": (
        "dedicated attention projection recurrent gate stack"
    ),
    "Attention Projection Recurrent Halting Options": (
        "attention projection recurrent halting controller"
    ),
    "Attention Projection Recurrent Halting Stack Options": (
        "dedicated attention projection recurrent halting stack"
    ),
    "Weight Generator Options": "dynamic weight generator",
    "Weight Generator Stack Options": "dedicated weight-generator stack",
    "Bias Generator Options": "dynamic bias generator",
    "Bias Generator Stack Options": "dedicated bias-generator stack",
    "Diagonal Generator Options": "dynamic diagonal generator",
    "Diagonal Generator Stack Options": "dedicated diagonal-generator stack",
    "Mask Options": "row or column mask controller",
    "Mask Stack Options": "dedicated mask-generator stack",
    "Input Boundary Model Options": "input boundary model",
    "Output Boundary Model Options": "output boundary model",
}

FLAG_DISABLED_PREFIXES = (
    "GATE_",
    "HALTING_",
    "MEMORY_",
    "RECURRENT_",
    "WEIGHT_",
    "BIAS_",
    "DIAGONAL_",
    "MASK_",
)

INHERITED_STACK_SECTIONS = {
    "Gate Stack Options",
    "Halting Stack Options",
    "Memory Stack Options",
    "Recurrent Gate Stack Options",
    "Recurrent Halting Stack Options",
    "Feed-Forward Gate Stack Options",
    "Feed-Forward Halting Stack Options",
    "Feed-Forward Memory Stack Options",
    "Feed-Forward Recurrent Gate Stack Options",
    "Feed-Forward Recurrent Halting Stack Options",
    "Attention Projection Gate Stack Options",
    "Attention Projection Halting Stack Options",
    "Attention Projection Memory Stack Options",
    "Attention Projection Recurrent Gate Stack Options",
    "Attention Projection Recurrent Halting Stack Options",
    "Weight Generator Stack Options",
    "Bias Generator Stack Options",
    "Diagonal Generator Stack Options",
    "Mask Stack Options",
}
