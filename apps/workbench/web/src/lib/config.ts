import type { ConfigField } from "@/lib/api/models";
import type { ConfigValue } from "@/lib/api/schemas";

export type OverrideValues = Record<string, string>;

export type ActiveOverrideScope = "preset" | "snapshot";

export type ConfigSection = {
  title: string;
  idPath?: string[];
  fields: ConfigField[];
  children?: ConfigSection[];
  controlFieldKey?: string | null;
};

export type ConfigFieldGroup = {
  title: string;
  fields: ConfigField[];
  controlField?: ConfigField;
  isEnabled?: boolean;
  disabledReason?: string;
  stackHint?: {
    label: string;
    title: string;
    sourceTitle: string;
  };
};

export type ConfigSearchOption = {
  sectionTitle: string;
  rootSectionKey: string;
  field: ConfigField;
  key: string;
  label: string;
  configKey: string;
  flag: string;
  type: string;
};

export type ConfigSearchState = {
  query: string;
  selectedFieldKey?: string | null;
};

export type InheritedStackSectionHint = {
  label: string;
  title: string;
  sourceTitle: string;
  isCustom: boolean;
};

export type InheritedConfigFieldHint = {
  label: string;
  title: string;
  sourceTitle: string;
  sourceField: ConfigField;
};

export function configKeyToken(key: string) {
  return key.trim().replace(/-/g, "_").toLowerCase();
}

function configFieldMatchesKey(field: ConfigField, key: string) {
  const token = configKeyToken(key);
  return (
    configKeyToken(field.key) === token ||
    configKeyToken(field.configKey) === token
  );
}

export function overrideValue(overrides: OverrideValues, key: string) {
  if (Object.prototype.hasOwnProperty.call(overrides, key)) {
    return overrides[key];
  }
  const token = configKeyToken(key);
  const entry = Object.entries(overrides).find(
    ([overrideKey]) => configKeyToken(overrideKey) === token,
  );
  return entry?.[1];
}

// Older backends send headings without canonical section metadata.
const LEGACY_SECTION_CONTROL_KEYS = new Map([
  ["Attention Mode", "expert_attention_flag"],
  ["Gate Options", "stack_gate_flag"],
  ["Gate Stack Options", "gate_stack_independent_flag"],
  ["Halting Options", "stack_halting_flag"],
  ["Halting Stack Options", "halting_stack_independent_flag"],
  ["Memory Options", "memory_flag"],
  ["Memory Stack Options", "memory_stack_independent_flag"],
  ["Router Gate Options", "router_stack_gate_flag"],
  ["Router Gate Stack Options", "router_gate_stack_independent_flag"],
  ["Router Halting Options", "router_stack_halting_flag"],
  ["Router Halting Stack Options", "router_halting_stack_independent_flag"],
  ["Router Memory Options", "router_memory_flag"],
  ["Router Memory Stack Options", "router_memory_stack_independent_flag"],
  ["Router Recurrent Layer Options", "router_recurrent_flag"],
  ["Router Recurrent Gate Options", "router_recurrent_stack_gate_flag"],
  [
    "Router Recurrent Gate Stack Options",
    "router_recurrent_gate_stack_independent_flag",
  ],
  ["Router Recurrent Halting Options", "router_recurrent_stack_halting_flag"],
  [
    "Router Recurrent Halting Stack Options",
    "router_recurrent_halting_stack_independent_flag",
  ],
  ["Expert Gate Options", "expert_stack_gate_flag"],
  ["Expert Gate Stack Options", "expert_gate_stack_independent_flag"],
  ["Expert Halting Options", "expert_stack_halting_flag"],
  ["Expert Halting Stack Options", "expert_halting_stack_independent_flag"],
  ["Expert Memory Options", "expert_memory_flag"],
  ["Expert Memory Stack Options", "expert_memory_stack_independent_flag"],
  ["Expert Recurrent Layer Options", "expert_recurrent_flag"],
  ["Expert Recurrent Gate Options", "expert_recurrent_stack_gate_flag"],
  [
    "Expert Recurrent Gate Stack Options",
    "expert_recurrent_gate_stack_independent_flag",
  ],
  ["Expert Recurrent Halting Options", "expert_recurrent_stack_halting_flag"],
  [
    "Expert Recurrent Halting Stack Options",
    "expert_recurrent_halting_stack_independent_flag",
  ],
  ["Feed-Forward Gate Options", "ff_stack_gate_flag"],
  ["Feed-Forward Gate Stack Options", "ff_gate_stack_independent_flag"],
  ["Feed-Forward Halting Options", "ff_stack_halting_flag"],
  ["Feed-Forward Halting Stack Options", "ff_halting_stack_independent_flag"],
  ["Feed-Forward Memory Options", "ff_memory_flag"],
  ["Feed-Forward Memory Stack Options", "ff_memory_stack_independent_flag"],
  ["Feed-Forward Recurrent Layer Options", "ff_recurrent_flag"],
  ["Feed-Forward Recurrent Gate Options", "ff_recurrent_stack_gate_flag"],
  [
    "Feed-Forward Recurrent Gate Stack Options",
    "ff_recurrent_gate_stack_independent_flag",
  ],
  ["Feed-Forward Recurrent Halting Options", "ff_recurrent_stack_halting_flag"],
  [
    "Feed-Forward Recurrent Halting Stack Options",
    "ff_recurrent_halting_stack_independent_flag",
  ],
  ["Attention Projection Gate Options", "attn_stack_gate_flag"],
  [
    "Attention Projection Gate Stack Options",
    "attn_gate_stack_independent_flag",
  ],
  ["Attention Projection Halting Options", "attn_stack_halting_flag"],
  [
    "Attention Projection Halting Stack Options",
    "attn_halting_stack_independent_flag",
  ],
  ["Attention Projection Memory Options", "attn_memory_flag"],
  [
    "Attention Projection Memory Stack Options",
    "attn_memory_stack_independent_flag",
  ],
  ["Attention Projection Recurrent Layer Options", "attn_recurrent_flag"],
  [
    "Attention Projection Recurrent Gate Options",
    "attn_recurrent_stack_gate_flag",
  ],
  [
    "Attention Projection Recurrent Gate Stack Options",
    "attn_recurrent_gate_stack_independent_flag",
  ],
  [
    "Attention Projection Recurrent Halting Options",
    "attn_recurrent_stack_halting_flag",
  ],
  [
    "Attention Projection Recurrent Halting Stack Options",
    "attn_recurrent_halting_stack_independent_flag",
  ],
  ["Recurrent Layer Options", "recurrent_flag"],
  ["Recurrent Gate Options", "recurrent_stack_gate_flag"],
  ["Recurrent Gate Stack Options", "recurrent_gate_stack_independent_flag"],
  ["Recurrent Halting Options", "recurrent_stack_halting_flag"],
  ["Recurrent Halting Stack Options", "recurrent_halting_stack_independent_flag"],
  ["Weight Options", "weight_option_flag"],
  ["Weight Generator Options", "weight_option_flag"],
  ["Weight Generator Stack Options", "weight_generator_stack_independent_flag"],
  ["Bias Options", "bias_option_flag"],
  ["Bias Generator Options", "bias_option_flag"],
  ["Bias Generator Stack Options", "bias_generator_stack_independent_flag"],
  ["Diagonal Options", "diagonal_option_flag"],
  ["Diagonal Generator Options", "diagonal_option_flag"],
  ["Diagonal Generator Stack Options", "diagonal_generator_stack_independent_flag"],
  ["Mask Options", "mask_option_flag"],
  ["Mask Stack Options", "mask_generator_stack_independent_flag"],
  ["Router Weight Generator Options", "router_weight_option_flag"],
  [
    "Router Weight Generator Stack Options",
    "router_weight_generator_stack_independent_flag",
  ],
  ["Router Bias Generator Options", "router_bias_option_flag"],
  [
    "Router Bias Generator Stack Options",
    "router_bias_generator_stack_independent_flag",
  ],
  ["Router Diagonal Generator Options", "router_diagonal_option_flag"],
  [
    "Router Diagonal Generator Stack Options",
    "router_diagonal_generator_stack_independent_flag",
  ],
  ["Router Mask Options", "router_mask_option_flag"],
  [
    "Router Mask Stack Options",
    "router_mask_generator_stack_independent_flag",
  ],
]);

const INHERITED_STACK_SECTIONS_BY_CONTROL = new Map([
  [
    "gate_stack_independent_flag",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "halting_stack_independent_flag",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "memory_stack_independent_flag",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "router_gate_stack_independent_flag",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "router_halting_stack_independent_flag",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "router_memory_stack_independent_flag",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "router_recurrent_gate_stack_independent_flag",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "router_recurrent_halting_stack_independent_flag",
    {
      sourceTitle: "Layer Stack Submodule Options",
      inheritedLabel: "Inherits Layer Stack Submodule",
    },
  ],
  [
    "expert_gate_stack_independent_flag",
    {
      sourceTitle: "Expert Stack Options",
      inheritedLabel: "Inherits Expert Stack",
    },
  ],
  [
    "expert_halting_stack_independent_flag",
    {
      sourceTitle: "Expert Stack Options",
      inheritedLabel: "Inherits Expert Stack",
    },
  ],
  [
    "expert_memory_stack_independent_flag",
    {
      sourceTitle: "Expert Stack Options",
      inheritedLabel: "Inherits Expert Stack",
    },
  ],
  [
    "expert_recurrent_gate_stack_independent_flag",
    {
      sourceTitle: "Expert Stack Options",
      inheritedLabel: "Inherits Expert Stack",
    },
  ],
  [
    "expert_recurrent_halting_stack_independent_flag",
    {
      sourceTitle: "Expert Stack Options",
      inheritedLabel: "Inherits Expert Stack",
    },
  ],
  [
    "ff_gate_stack_independent_flag",
    {
      sourceTitle: "Feed-Forward Stack Options",
      inheritedLabel: "Inherits Feed-Forward Stack",
    },
  ],
  [
    "ff_halting_stack_independent_flag",
    {
      sourceTitle: "Feed-Forward Stack Options",
      inheritedLabel: "Inherits Feed-Forward Stack",
    },
  ],
  [
    "ff_memory_stack_independent_flag",
    {
      sourceTitle: "Feed-Forward Stack Options",
      inheritedLabel: "Inherits Feed-Forward Stack",
    },
  ],
  [
    "ff_recurrent_gate_stack_independent_flag",
    {
      sourceTitle: "Feed-Forward Stack Options",
      inheritedLabel: "Inherits Feed-Forward Stack",
    },
  ],
  [
    "ff_recurrent_halting_stack_independent_flag",
    {
      sourceTitle: "Feed-Forward Stack Options",
      inheritedLabel: "Inherits Feed-Forward Stack",
    },
  ],
  [
    "attn_gate_stack_independent_flag",
    {
      sourceTitle: "Attention Projection Stack Options",
      inheritedLabel: "Inherits Attention Projection Stack",
    },
  ],
  [
    "attn_halting_stack_independent_flag",
    {
      sourceTitle: "Attention Projection Stack Options",
      inheritedLabel: "Inherits Attention Projection Stack",
    },
  ],
  [
    "attn_memory_stack_independent_flag",
    {
      sourceTitle: "Attention Projection Stack Options",
      inheritedLabel: "Inherits Attention Projection Stack",
    },
  ],
  [
    "attn_recurrent_gate_stack_independent_flag",
    {
      sourceTitle: "Attention Projection Stack Options",
      inheritedLabel: "Inherits Attention Projection Stack",
    },
  ],
  [
    "attn_recurrent_halting_stack_independent_flag",
    {
      sourceTitle: "Attention Projection Stack Options",
      inheritedLabel: "Inherits Attention Projection Stack",
    },
  ],
  [
    "recurrent_gate_stack_independent_flag",
    {
      sourceTitle: "Gate Stack Options",
      inheritedLabel: "Inherits Gate Stack",
    },
  ],
  [
    "recurrent_halting_stack_independent_flag",
    {
      sourceTitle: "Halting Stack Options",
      inheritedLabel: "Inherits Halting Stack",
    },
  ],
  [
    "weight_generator_stack_independent_flag",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "bias_generator_stack_independent_flag",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "diagonal_generator_stack_independent_flag",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "mask_generator_stack_independent_flag",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "router_weight_generator_stack_independent_flag",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "router_bias_generator_stack_independent_flag",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "router_diagonal_generator_stack_independent_flag",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
  [
    "router_mask_generator_stack_independent_flag",
    {
      sourceTitle: "Adaptive Generator Stack Options",
      inheritedLabel: "Inherits Adaptive Stack",
    },
  ],
]);

const GENERAL_CONFIG_SECTION = "General";
const GLOBAL_CONFIG_SECTION = "Global";
const LAYER_STACK_OPTIONS_SECTION = "Layer Stack Options";
const LAYER_STACK_SUBMODULE_OPTIONS_SECTION = "Layer Stack Submodule Options";
const INPUT_BOUNDARY_MODEL_SECTION = "Input Boundary Model Options";
const OUTPUT_BOUNDARY_MODEL_SECTION = "Output Boundary Model Options";
const ADAPTIVE_GENERATOR_STACK_SECTION = "Adaptive Generator Stack Options";
const MIXTURE_OF_EXPERTS_SECTION = "Mixture Of Experts Model Options";
const ATTENTION_MODE_SECTION = "Attention Mode";

const MIXTURE_OF_EXPERTS_ID = "mixture_of_experts_model";
const ATTENTION_MODE_ID = "attention_mode";
const ATTENTION_MODE_FIELDS = new Set([
  "expert_attention_flag",
  "expert_attention_use_kv_expert_models_flag",
]);

const DISPLAY_CONFIG_FIELD_LABELS_BY_KEY = new Map<string, string>([
  ["expert_attention_flag", "Use MixtureOfAttentionHeads"],
  ["expert_attention_use_kv_expert_models_flag", "Expert K/V projections"],
]);

const DISPLAY_CONFIG_SECTIONS = new Map<
  string,
  { title: string; description: string }
>([
  [
    LAYER_STACK_OPTIONS_SECTION,
    {
      title: "Layer Hidden Stack Options",
      description: "Hidden stack inside each layer",
    },
  ],
  [
    LAYER_STACK_SUBMODULE_OPTIONS_SECTION,
    {
      title: "Shared Submodule Stack Defaults",
      description:
        "Defaults for gate, halting, memory, and recurrent stacks unless overridden.",
    },
  ],
  [
    ATTENTION_MODE_SECTION,
    {
      title: "Attention Mode",
      description:
        "Off uses SelfAttention. On uses MixtureOfAttentionHeads for expert-routed attention projections.",
    },
  ],
]);

export function displayConfigFieldSection(field: ConfigField) {
  return field.section || GENERAL_CONFIG_SECTION;
}

export function displayConfigSectionTitle(title: string) {
  return DISPLAY_CONFIG_SECTIONS.get(title)?.title ?? title;
}

export function displayConfigSectionDescription(title: string) {
  return DISPLAY_CONFIG_SECTIONS.get(title)?.description;
}

function displaySectionField(field: ConfigField): ConfigField {
  if (!ATTENTION_MODE_FIELDS.has(configKeyToken(field.key))) {
    return field;
  }
  const rootTitle = field.sectionIdPath?.[0] === MIXTURE_OF_EXPERTS_ID
    ? field.sectionPath[0]!
    : MIXTURE_OF_EXPERTS_SECTION;
  return {
    ...field,
    section: ATTENTION_MODE_SECTION,
    sectionPath: [rootTitle, ATTENTION_MODE_SECTION],
    ...(field.sectionIdPath && {
      sectionIdPath: [MIXTURE_OF_EXPERTS_ID, ATTENTION_MODE_ID],
      sectionControlFieldKey: "expert_attention_flag",
    }),
  };
}

function sentenceCaseLabel(label: string) {
  const trimmed = label.trim().replace(/\s+/g, " ");
  return trimmed ? `${trimmed[0]!.toUpperCase()}${trimmed.slice(1)}` : trimmed;
}

function sectionTitleLabelPrefixes(sectionTitle: string) {
  const prefix = sectionTitle
    .trim()
    .replace(/\s+options$/i, "")
    .toLowerCase();
  const abbreviatedPrefix = prefix
    .replace(/\bfeed-forward\b/g, "ff")
    .replace(/\battention projection\b/g, "attn");

  return [
    prefix,
    abbreviatedPrefix,
    prefix.replace(/\blayer\b\s*/g, "").trim(),
    abbreviatedPrefix.replace(/\blayer\b\s*/g, "").trim(),
  ].filter((candidate, index, candidates) =>
    candidate.length > 0 && candidates.indexOf(candidate) === index
  );
}

function stackFieldLabelPrefix(fieldKey: string) {
  const key = configKeyToken(fieldKey);
  if (key.startsWith("stack_")) {
    return "stack";
  }
  const stackIndex = key.indexOf("_stack_");
  if (stackIndex === -1) {
    return undefined;
  }
  return key
    .slice(0, stackIndex + "_stack".length)
    .replace(/_/g, " ");
}

function stackSectionLabelPrefix(sectionTitle: string) {
  if (sectionTitle === LAYER_STACK_OPTIONS_SECTION) {
    return "stack";
  }
  if (sectionTitle === LAYER_STACK_SUBMODULE_OPTIONS_SECTION) {
    return "submodule stack";
  }
  return undefined;
}

export function displayConfigFieldLabel(
  field: ConfigField,
  section: ConfigSection,
  groupTitle?: string,
) {
  const explicitLabel = DISPLAY_CONFIG_FIELD_LABELS_BY_KEY.get(
    configKeyToken(field.key),
  );
  if (explicitLabel) {
    return explicitLabel;
  }

  const controlFieldKey = sectionControlKey(section);
  if (
    controlFieldKey &&
    configFieldMatchesKey(field, controlFieldKey)
  ) {
    return configKeyToken(field.key).endsWith("_stack_independent_flag")
      ? "Use custom stack"
      : "Enabled";
  }

  const label = field.label.trim().replace(/\s+/g, " ");
  const normalizedLabel = label.toLowerCase();
  const prefixCandidates = [
    stackFieldLabelPrefix(field.key),
    stackSectionLabelPrefix(section.title),
    ...sectionTitleLabelPrefixes(section.title),
    groupTitle ? groupTitle.toLowerCase() : undefined,
  ]
    .filter((prefix): prefix is string => Boolean(prefix))
    .sort((left, right) => right.length - left.length);

  for (const prefix of prefixCandidates) {
    if (normalizedLabel.startsWith(`${prefix} `)) {
      return sentenceCaseLabel(label.slice(prefix.length + 1));
    }
  }

  return sentenceCaseLabel(label);
}

export function normalizeConfigFieldForDisplay(field: ConfigField): ConfigField {
  const section = displayConfigFieldSection(field);
  return field.section === section ? field : { ...field, section };
}

function normalizedFieldSectionPath(field: ConfigField) {
  const sectionPath = field.sectionPath.filter(
    (section): section is string => typeof section === "string" && section.length > 0,
  );
  if (sectionPath.length === 0) {
    throw new Error(`Config field ${field.key} is missing sectionPath metadata.`);
  }
  return sectionPath;
}

export function configSectionFields(section: ConfigSection): ConfigField[] {
  return [
    ...section.fields,
    ...(section.children ?? []).flatMap((child) => configSectionFields(child)),
  ];
}

export function configSectionsFields(sections: ConfigSection[]): ConfigField[] {
  return sections.flatMap((section) => configSectionFields(section));
}

export function configSectionKey(section: Pick<ConfigSection, "title" | "idPath">) {
  return section.idPath?.join("/") ?? section.title;
}

export function groupConfigFieldsBySectionPath(fields: ConfigField[]): ConfigSection[] {
  type DraftSection = ConfigSection & {
    children: DraftSection[];
  };

  const rootSections: DraftSection[] = [];

  function getOrCreateChildSection(
    parent: DraftSection[] | DraftSection,
    title: string,
    idPath?: string[],
  ) {
    const children = Array.isArray(parent) ? parent : parent.children;
    const identity = configSectionKey({ title, idPath });
    let child = children.find((candidate) => configSectionKey(candidate) === identity);
    if (!child) {
      child = { title, idPath, fields: [], children: [] };
      children.push(child);
    }
    return child;
  }

  for (const rawField of fields) {
    const field = displaySectionField(normalizeConfigFieldForDisplay(rawField));
    const sectionPath = normalizedFieldSectionPath(field);
    const idPath = field.sectionIdPath?.length === sectionPath.length
      ? field.sectionIdPath
      : undefined;
    let currentSection = getOrCreateChildSection(
      rootSections,
      sectionPath[0]!,
      idPath?.slice(0, 1),
    );
    for (const [index, sectionTitle] of sectionPath.slice(1).entries()) {
      currentSection = getOrCreateChildSection(
        currentSection,
        sectionTitle,
        idPath?.slice(0, index + 2),
      );
    }
    if (idPath && field.sectionControlFieldKey !== undefined) {
      currentSection.controlFieldKey = field.sectionControlFieldKey;
    }
    currentSection.fields.push(field);
  }

  function finalizeSection(section: DraftSection): ConfigSection {
    return {
      title: section.title,
      fields: section.fields,
      ...(section.idPath && {
        idPath: section.idPath,
        controlFieldKey: section.controlFieldKey ?? null,
      }),
      ...(section.children.length > 0
        ? { children: section.children.map(finalizeSection) }
        : {}),
    };
  }

  return rootSections.map(finalizeSection);
}

export function fieldValue(field: ConfigField, overrides: OverrideValues) {
  const value = field.locked
    ? field.lockedValue
    : (overrideValue(overrides, field.key) ?? field.default);
  return value === null || value === undefined ? "" : String(value);
}

function normalizePrimitiveConfigValue(value: ConfigValue | string) {
  return value === null || value === undefined ? "" : String(value).trim();
}

export function normalizeConfigFieldValue(
  field: ConfigField,
  value: ConfigValue | string,
) {
  const raw = normalizePrimitiveConfigValue(value);
  if (field.nullable && raw === "") {
    return "null";
  }
  if (field.type === "bool") {
    const lower = raw.toLowerCase();
    if (lower === "true" || lower === "false") {
      return lower;
    }
  }
  if (field.type === "int") {
    const numberValue = Number(raw);
    if (Number.isInteger(numberValue)) {
      return String(numberValue);
    }
  }
  if (field.type === "float") {
    const numberValue = Number(raw);
    if (Number.isFinite(numberValue)) {
      return String(numberValue);
    }
  }
  return raw;
}

export function defaultConfigFieldValue(field: ConfigField) {
  return normalizeConfigFieldValue(field, field.default);
}

export function overrideValueForConfigField(
  field: ConfigField,
  value: ConfigValue | string,
) {
  const normalizedValue = normalizeConfigFieldValue(field, value);
  return field.nullable && normalizedValue === "null" ? "" : normalizedValue;
}

export function hasOverride(overrides: OverrideValues, key: string) {
  return overrideValue(overrides, key) !== undefined;
}

export const ADAPTIVE_OPTION_PAIRS = [
  { flagKey: "weight_option_flag", optionKey: "weight_option" },
  { flagKey: "bias_option_flag", optionKey: "bias_option" },
  { flagKey: "diagonal_option_flag", optionKey: "diagonal_option" },
  { flagKey: "mask_option_flag", optionKey: "row_mask_option" },
  { flagKey: "router_weight_option_flag", optionKey: "router_weight_option" },
  { flagKey: "router_bias_option_flag", optionKey: "router_bias_option" },
  { flagKey: "router_diagonal_option_flag", optionKey: "router_diagonal_option" },
  { flagKey: "router_mask_option_flag", optionKey: "router_row_mask_option" },
] as const;

export function configFieldByKey(fields: ConfigField[], key: string) {
  return fields.find((field) => configFieldMatchesKey(field, key));
}

export function inheritedHiddenModelFieldHint(
  section: ConfigSection,
  allFields: ConfigField[],
): InheritedConfigFieldHint | undefined {
  const isLayerStack = section.idPath
    ? section.idPath.at(-1) === "stack_num_layers"
    : section.title === LAYER_STACK_OPTIONS_SECTION;
  if (!isLayerStack) {
    return undefined;
  }
  if (configFieldByKey(section.fields, "hidden_dim")) {
    return undefined;
  }
  const sourceField = configFieldByKey(allFields, "hidden_dim");
  if (!sourceField) {
    return undefined;
  }
  const sourceTitle = sourceField.section || GLOBAL_CONFIG_SECTION;
  return {
    label: sourceField.label,
    title: `${sourceField.label} comes from ${sourceTitle} / ${sourceField.configKey}.`,
    sourceTitle,
    sourceField,
  };
}

export function concreteConfigOptionChoice(field: ConfigField) {
  for (const choice of field.choices) {
    if (choice === null || choice === undefined) {
      continue;
    }
    const value = String(choice).trim();
    const normalized = value.toLowerCase();
    if (value && normalized !== "none" && normalized !== "null") {
      return value;
    }
  }
  return undefined;
}

export function isPresentConfigOptionValue(value: string | undefined) {
  const normalized = (value ?? "").trim().toLowerCase();
  return normalized !== "" && normalized !== "null" && normalized !== "none";
}

function adaptiveFlagKeyForOptionField(fieldKey: string) {
  const fieldToken = configKeyToken(fieldKey);
  return ADAPTIVE_OPTION_PAIRS.find(
    ({ optionKey }) => configKeyToken(optionKey) === fieldToken,
  )?.flagKey;
}

export function isBoundaryModelControlField(field: ConfigField) {
  const key = configKeyToken(field.key);
  return (
    key === "input_layer_weight_option" ||
    key === "input_layer_bias_option" ||
    key === "input_layer_diagonal_option" ||
    key === "input_layer_row_mask_option" ||
    key === "output_layer_weight_option" ||
    key === "output_layer_bias_option" ||
    key === "output_layer_diagonal_option" ||
    key === "output_layer_row_mask_option"
  );
}

export function isAdaptiveOptionNoneSuppressed(
  field: ConfigField,
  overrides: OverrideValues,
) {
  if (isBoundaryModelControlField(field)) {
    return isPresentConfigOptionValue(fieldValue(field, overrides));
  }

  const flagKey = adaptiveFlagKeyForOptionField(field.key);
  if (!flagKey) {
    return false;
  }
  return isEnabledConfigValue(overrideValue(overrides, flagKey) ?? "");
}

export function configFieldSelectOptions(
  field: ConfigField,
  overrides: OverrideValues,
) {
  const includeNone = field.nullable && !isAdaptiveOptionNoneSuppressed(field, overrides);
  return [
    ...(includeNone ? [{ value: "", label: "None" }] : []),
    ...field.choices.map((choice) => ({
      value: String(choice),
      label: String(choice),
    })),
  ];
}

export function overrideDigest(overrides: OverrideValues) {
  const entries = Object.keys(overrides)
    .sort()
    .map((key) => [key, overrides[key]]);
  return JSON.stringify(entries);
}

export function activeOverrideScopeLabel(scope: ActiveOverrideScope) {
  return scope === "snapshot" ? "Snapshot draft" : "Preset overrides";
}

export function isEnabledConfigValue(value: string) {
  return ["true", "1", "yes", "on"].includes(value.trim().toLowerCase());
}

function sectionControlKey(section: ConfigSection) {
  return section.controlFieldKey === undefined
    ? LEGACY_SECTION_CONTROL_KEYS.get(section.title)
    : section.controlFieldKey;
}

export function controlledSectionFlagField(section: ConfigSection) {
  const controlFieldKey = sectionControlKey(section);
  if (controlFieldKey) {
    return section.fields.find((field) => configFieldMatchesKey(field, controlFieldKey));
  }
  return undefined;
}

export function controlledSectionDisabledReason(
  section: ConfigSection,
  controlField: ConfigField,
) {
  return `Enable ${controlField.label} before editing ${section.title}.`;
}

export function controlledSectionState(
  section: ConfigSection,
  overrides: OverrideValues,
) {
  const controlField = controlledSectionFlagField(section);
  const isControlled = Boolean(controlField);
  const isEnabled = controlField
    ? isEnabledConfigValue(fieldValue(controlField, overrides))
    : true;

  return {
    controlField,
    isControlled,
    isEnabled,
    disabledReason:
      controlField && !isEnabled
        ? controlledSectionDisabledReason(section, controlField)
        : undefined,
  };
}

export function inheritedStackSectionHint(
  section: ConfigSection,
  overrides: OverrideValues,
): InheritedStackSectionHint | undefined {
  const controlField = controlledSectionFlagField(section);
  if (!controlField) {
    return undefined;
  }
  const inheritance = INHERITED_STACK_SECTIONS_BY_CONTROL.get(configKeyToken(controlField.key));
  if (!inheritance) {
    return undefined;
  }

  const isCustom = isEnabledConfigValue(fieldValue(controlField, overrides));
  const title = isCustom
    ? `Uses ${section.title} values while ${controlField.label} is on. Disable it to inherit ${inheritance.sourceTitle}.`
    : `Uses ${inheritance.sourceTitle} while ${controlField.label} is off. Enable it to use ${section.title} values.`;

  return {
    label: isCustom ? "Custom Stack" : inheritance.inheritedLabel,
    title,
    sourceTitle: inheritance.sourceTitle,
    isCustom,
  };
}

export function disabledConfigFieldReasons(
  sections: ConfigSection[],
  overrides: OverrideValues,
) {
  const disabledReasons = new Map<string, string>();

  function collectDisabledReasons(
    section: ConfigSection,
    inheritedReason?: string,
  ) {
    const state = controlledSectionState(section, overrides);
    const ownDisabledReason =
      !state.isEnabled && state.disabledReason ? state.disabledReason : undefined;
    const sectionDisabledReason =
      inheritedReason ?? ownDisabledReason;
    const shouldPropagateOwnDisabledReason =
      !state.controlField ||
      !configKeyToken(state.controlField.key).endsWith("_stack_independent_flag");

    if (sectionDisabledReason) {
      for (const field of section.fields) {
        const isEditableControl =
          !inheritedReason &&
          state.controlField !== undefined &&
          configFieldMatchesKey(field, state.controlField.key);
        if (!isEditableControl) {
          disabledReasons.set(field.key, sectionDisabledReason);
        }
      }
    }

    if (!sectionDisabledReason) {
      const boundaryGroups = boundaryModelFieldGroups(
        section,
        section.fields,
        overrides,
      );
      for (const group of boundaryGroups ?? []) {
        if (!group.disabledReason || !group.controlField) {
          continue;
        }
        for (const field of group.fields) {
          if (!configFieldMatchesKey(field, group.controlField.key)) {
            disabledReasons.set(field.key, group.disabledReason);
          }
        }
      }
    }

    for (const child of section.children ?? []) {
      collectDisabledReasons(
        child,
        inheritedReason ??
          (shouldPropagateOwnDisabledReason ? ownDisabledReason : undefined),
      );
    }
  }

  for (const section of deriveNestedConfigSections(sections)) {
    collectDisabledReasons(section);
  }

  return disabledReasons;
}

export function fieldsByKey(sections: ConfigSection[]) {
  const fields = new Map<string, ConfigField>();
  for (const field of configSectionsFields(sections)) {
    fields.set(field.key, field);
  }
  return fields;
}

export function flattenConfigSearchOptions(
  sections: ConfigSection[],
) {
  const options: ConfigSearchOption[] = [];
  function collectSectionOptions(section: ConfigSection, rootSectionKey: string) {
    for (const field of section.fields) {
      options.push({
        sectionTitle: section.title,
        rootSectionKey,
        field,
        key: field.key,
        label: field.label,
        configKey: field.configKey,
        flag: field.flag,
        type: field.type,
      });
    }
    for (const child of section.children ?? []) {
      collectSectionOptions(child, rootSectionKey);
    }
  }
  for (const section of sections) {
    collectSectionOptions(section, configSectionKey(section));
  }
  return options;
}

function normalizedSearchText(value: string) {
  return value.trim().toLowerCase();
}

export function configSearchOptionMatchesQuery(
  option: Pick<
    ConfigSearchOption,
    "sectionTitle" | "key" | "label" | "configKey" | "flag"
  >,
  query: string,
) {
  const normalizedQuery = normalizedSearchText(query);
  if (!normalizedQuery) {
    return true;
  }

  return [
    option.label,
    option.key,
    option.configKey,
    option.flag,
    option.sectionTitle,
  ].some((value) => value.toLowerCase().includes(normalizedQuery));
}

export function filterConfigSectionsForSearch(
  sections: ConfigSection[],
  { query, selectedFieldKey }: ConfigSearchState,
) {
  const normalizedQuery = normalizedSearchText(query);
  const selectedKey = selectedFieldKey?.trim();
  if (!normalizedQuery && !selectedKey) {
    return sections;
  }

  function filterSection(section: ConfigSection): ConfigSection | undefined {
    const fields = section.fields.filter((field) => {
      if (selectedKey) {
        return configKeyToken(field.key) === configKeyToken(selectedKey);
      }

      return configSearchOptionMatchesQuery(
        {
          sectionTitle: section.title,
          key: field.key,
          label: field.label,
          configKey: field.configKey,
          flag: field.flag,
        },
        normalizedQuery,
      );
    });
    const children = (section.children ?? [])
      .map(filterSection)
      .filter((child): child is ConfigSection => Boolean(child));
    if (fields.length === 0 && children.length === 0) {
      return undefined;
    }
    return {
      ...section,
      fields,
      ...(children.length > 0 ? { children } : { children: undefined }),
    };
  }

  return sections
    .map(filterSection)
    .filter((section): section is ConfigSection => Boolean(section));
}

function boundaryModelPrefix(section: ConfigSection) {
  const anchor = section.idPath?.at(-1);
  if (anchor === "input_layer_weight_option" || (!anchor && section.title === INPUT_BOUNDARY_MODEL_SECTION)) {
    return "input_layer_";
  }
  if (anchor === "output_layer_weight_option" || (!anchor && section.title === OUTPUT_BOUNDARY_MODEL_SECTION)) {
    return "output_layer_";
  }
  return undefined;
}

function boundaryModelGroupTitle(prefix: string, fieldKey: string) {
  const key = configKeyToken(fieldKey);
  if (key.startsWith(`${prefix}weight_`)) {
    return "Weight";
  }
  if (key.startsWith(`${prefix}bias_`)) {
    return "Bias";
  }
  if (key.startsWith(`${prefix}diagonal_`)) {
    return "Diagonal";
  }
  if (
    key === `${prefix}row_mask_option` ||
    key.startsWith(`${prefix}mask_`)
  ) {
    return "Mask";
  }
  return undefined;
}

function boundaryModelControlFieldKey(prefix: string, groupTitle: string) {
  if (groupTitle === "Weight") {
    return `${prefix}weight_option`;
  }
  if (groupTitle === "Bias") {
    return `${prefix}bias_option`;
  }
  if (groupTitle === "Diagonal") {
    return `${prefix}diagonal_option`;
  }
  if (groupTitle === "Mask") {
    return `${prefix}row_mask_option`;
  }
  return undefined;
}

function boundaryModelStackHint(): NonNullable<ConfigFieldGroup["stackHint"]> {
  return {
    label: "Uses Adaptive Stack",
    title: `Boundary generator stack settings come from ${ADAPTIVE_GENERATOR_STACK_SECTION}.`,
    sourceTitle: ADAPTIVE_GENERATOR_STACK_SECTION,
  };
}

export function boundaryModelFieldGroups(
  section: ConfigSection,
  fields: ConfigField[],
  overrides: OverrideValues = {},
): ConfigFieldGroup[] | undefined {
  const prefix = boundaryModelPrefix(section);
  if (!prefix) {
    return undefined;
  }

  const groupsByTitle = new Map(
    [
      "Weight",
      "Bias",
      "Diagonal",
      "Mask",
    ].map((title) => [title, [] as ConfigField[]]),
  );

  for (const field of fields) {
    const groupTitle = boundaryModelGroupTitle(prefix, field.key);
    if (groupTitle) {
      groupsByTitle.get(groupTitle)?.push(field);
    }
  }

  const groups = Array.from(groupsByTitle, ([title, groupFields]) => {
    const controlFieldKey = boundaryModelControlFieldKey(prefix, title);
    const controlField = groupFields.find((field) =>
      configFieldMatchesKey(field, controlFieldKey ?? ""),
    );
    const isEnabled = controlField
      ? isPresentConfigOptionValue(fieldValue(controlField, overrides))
      : true;
    return {
      title,
      fields: groupFields,
      controlField,
      isEnabled,
      stackHint: boundaryModelStackHint(),
      disabledReason:
        controlField && !isEnabled
          ? `Select ${controlField.label} before editing ${title.toLowerCase()} boundary settings.`
          : undefined,
    };
  }).filter((group) => group.fields.length > 0);
  const groupedFieldCount = groups.reduce(
    (count, group) => count + group.fields.length,
    0,
  );

  return groupedFieldCount === fields.length ? groups : undefined;
}

function withSectionControlField(section: ConfigSection): ConfigSection {
  const controlFieldKey = sectionControlKey(section);
  const children = section.children?.map(withSectionControlField);
  return {
    ...section,
    ...(controlFieldKey ? { controlFieldKey } : {}),
    ...(children ? { children } : {}),
  };
}

function controlFieldForSection(section: ConfigSection | undefined) {
  if (!section?.controlFieldKey) {
    return undefined;
  }
  return section.fields.find((field) =>
    configFieldMatchesKey(field, section.controlFieldKey ?? ""),
  );
}

function withRequiredControlFields(
  section: ConfigSection,
  sourceSection: ConfigSection | undefined,
): ConfigSection {
  const controlField = controlFieldForSection(section) ?? controlFieldForSection(sourceSection);
  const hasControlField =
    controlField && section.fields.some((field) => field.key === controlField.key);
  const sourceChildren = new Map(
    (sourceSection?.children ?? []).map((child) => [configSectionKey(child), child]),
  );
  const children = (section.children ?? []).map((child) =>
    withRequiredControlFields(child, sourceChildren.get(configSectionKey(child))),
  );
  const fieldsWithControl =
    controlField && !hasControlField ? [controlField, ...section.fields] : section.fields;

  return {
    ...section,
    controlFieldKey: section.controlFieldKey ?? sourceSection?.controlFieldKey,
    fields: fieldsWithControl,
    children,
  };
}

export function deriveNestedConfigSections(
  sections: ConfigSection[],
  sourceSections?: ConfigSection[],
) {
  const derivedSections = groupConfigFieldsBySectionPath(
    configSectionsFields(sections),
  ).map(withSectionControlField);
  if (!sourceSections) {
    return derivedSections;
  }

  const sourceSectionsByKey = new Map(
    groupConfigFieldsBySectionPath(configSectionsFields(sourceSections))
      .map(withSectionControlField)
      .map((section) => [
        configSectionKey(section),
        section,
      ]),
  );

  return derivedSections.map((section) =>
    withRequiredControlFields(section, sourceSectionsByKey.get(configSectionKey(section))),
  );
}

export function sectionElementId(
  index: number,
  title: string,
  prefix = "full-config-section",
) {
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return `${prefix}-${index}-${slug || "section"}`;
}

export function sectionCountLabel(count: number, noun: "field" | "override" | "preset") {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

export function splitSectionSignature(signature: string) {
  return signature ? signature.split("\u0000") : [];
}
