import { type ConfigField } from "@/lib/api/models";
import {
  type ConfigSearchState,
  type ConfigSection,
  type InheritedStackSectionHint,
  type OverrideValues,
  boundaryModelFieldGroups,
  configFieldSelectOptions,
  configKeyToken,
  configSearchOptionMatchesQuery,
  configSectionFields,
  configSectionKey,
  configSectionsFields,
  concreteConfigOptionChoice,
  controlledSectionState,
  deriveNestedConfigSections,
  disabledConfigFieldReasons,
  displayConfigFieldLabel,
  displayConfigSectionDescription,
  displayConfigSectionTitle,
  fieldValue,
  filterConfigSectionsForSearch,
  flattenConfigSearchOptions,
  groupConfigFieldsBySectionPath,
  hasOverride,
  inheritedHiddenModelFieldHint,
  inheritedStackSectionHint,
  isEnabledConfigValue,
  isPresentConfigOptionValue,
  overrideValue,
  sectionElementId,
} from "@/lib/config";
import {
  applicableConfigFields,
  runtimeDefaultsMetrics,
  type RuntimeDefaultsMetrics,
} from "@/features/workbench/state/runtime-defaults/runtime-defaults";

export type RuntimeDefaultsFieldPresentation = {
  schema: ConfigField;
  key: string;
  label: string;
  value: string;
  selectOptions: Array<{ value: string; label: string }>;
  isModified: boolean;
  isPresetOwned: boolean;
  isEnabledValue: boolean;
  modeLabel?: string;
  disabledReason?: string;
};

export type RuntimeDefaultsInheritedFieldPresentation = {
  label: string;
  title: string;
  sourceTitle: string;
  field: RuntimeDefaultsFieldPresentation;
};

export type RuntimeDefaultsFieldGroupPresentation = {
  id: string;
  title: string;
  fields: RuntimeDefaultsFieldPresentation[];
  controlField?: RuntimeDefaultsFieldPresentation;
  isEnabled: boolean;
  isSwitchDisabled: boolean;
  firstConcreteOption?: string;
  disabledReason?: string;
  stackHint?: {
    label: string;
    title: string;
    sourceTitle: string;
  };
  metrics: RuntimeDefaultsMetrics;
};

export type RuntimeDefaultsSectionPresentation = {
  id: string;
  key: string;
  title: string;
  displayTitle: string;
  displayDescription?: string;
  fields: RuntimeDefaultsFieldPresentation[];
  bodyFields: RuntimeDefaultsFieldPresentation[];
  fieldGroups?: RuntimeDefaultsFieldGroupPresentation[];
  children: RuntimeDefaultsSectionPresentation[];
  controlField?: RuntimeDefaultsFieldPresentation;
  disabledReason?: string;
  isDisabled: boolean;
  directMetrics: RuntimeDefaultsMetrics;
  treeMetrics: RuntimeDefaultsMetrics;
  stackInheritanceHint?: InheritedStackSectionHint;
  inheritedField?: RuntimeDefaultsInheritedFieldPresentation;
  childKeySignature: string;
  groupTitleSignature: string;
  enabledGroupTitleSignature: string;
};

export type RuntimeDefaultsSearchOptionPresentation = {
  sectionTitle: string;
  rootSectionKey: string;
  field: RuntimeDefaultsFieldPresentation;
  key: string;
  label: string;
  configKey: string;
  flag: string;
  type: string;
};

export type RuntimeDefaultsSchemaPresentation = {
  schemaFields: ConfigField[];
  fieldCount: number;
  presetOwnedFieldCount: number;
  sections: RuntimeDefaultsSectionPresentation[];
  defaultOpenSectionKeys: string[];
  searchOpenKey: string;
  isSearchActive: boolean;
  search: {
    options: RuntimeDefaultsSearchOptionPresentation[];
    matchesQuery: (
      option: RuntimeDefaultsSearchOptionPresentation,
      query: string,
    ) => boolean;
  };
};

function shouldPresentConfigField(field: ConfigField, overrides: OverrideValues) {
  if (field.type !== "json" || field.legacyJson !== true) {
    return true;
  }
  return (
    isPresentConfigOptionValue(String(field.default ?? "")) ||
    (field.locked === true &&
      isPresentConfigOptionValue(String(field.lockedValue ?? ""))) ||
    isPresentConfigOptionValue(overrideValue(overrides, field.key))
  );
}

function attentionModeValueLabel(field: ConfigField, value: string) {
  if (configKeyToken(field.key) !== "expert_attention_flag") {
    return undefined;
  }
  return isEnabledConfigValue(value)
    ? "MixtureOfAttentionHeads"
    : "SelfAttention";
}

function presentField(
  field: ConfigField,
  overrides: OverrideValues,
  label: string,
  disabledReason?: string,
): RuntimeDefaultsFieldPresentation {
  const value = fieldValue(field, overrides);
  return {
    schema: field,
    key: field.key,
    label,
    value,
    selectOptions: configFieldSelectOptions(field, overrides),
    isModified: hasOverride(overrides, field.key),
    isPresetOwned: field.locked === true,
    isEnabledValue: isEnabledConfigValue(value),
    modeLabel: attentionModeValueLabel(field, value),
    disabledReason,
  };
}

function descendantFieldKeys(sections: ConfigSection[]) {
  return new Set(
    sections.flatMap((section) =>
      configSectionFields(section).map((field) => field.key),
    ),
  );
}

function presentSection({
  section,
  sourceSection,
  allFields,
  disabledFieldReasonByKey,
  overrides,
  showInheritedFields,
  inheritedDisabledReason,
  index,
  idPrefix,
}: {
  section: ConfigSection;
  sourceSection: ConfigSection;
  allFields: ConfigField[];
  disabledFieldReasonByKey: Map<string, string>;
  overrides: OverrideValues;
  showInheritedFields: boolean;
  inheritedDisabledReason?: string;
  index: number;
  idPrefix?: string;
}): RuntimeDefaultsSectionPresentation {
  const key = configSectionKey(section);
  const id = section.idPath
    ? `full-config-section-${key}`
    : sectionElementId(index, section.title, idPrefix);
  const controlState = controlledSectionState(sourceSection, overrides);
  const disabledReason = inheritedDisabledReason ?? controlState.disabledReason;
  const controlField = controlState.controlField
    ? presentField(
        controlState.controlField,
        overrides,
        displayConfigFieldLabel(controlState.controlField, sourceSection),
        disabledFieldReasonByKey.get(controlState.controlField.key),
      )
    : undefined;
  const childFieldKeys = descendantFieldKeys(section.children ?? []);
  const bodySchemas = section.fields.filter(
    (field) => field.key !== controlField?.key && !childFieldKeys.has(field.key),
  );
  const boundaryFieldGroups = boundaryModelFieldGroups(
    section,
    bodySchemas,
    overrides,
  );
  const fieldGroups = boundaryFieldGroups?.map((group, groupIndex) => {
    const groupControlField = group.controlField
      ? presentField(
          group.controlField,
          overrides,
          displayConfigFieldLabel(group.controlField, section, group.title),
          disabledFieldReasonByKey.get(group.controlField.key),
        )
      : undefined;
    const firstConcreteOption = group.controlField
      ? concreteConfigOptionChoice(group.controlField)
      : undefined;
    const isEnabled = group.isEnabled === true;
    return {
      id: sectionElementId(groupIndex, group.title, `${id}-boundary-group`),
      title: group.title,
      fields: group.fields.map((field) =>
        presentField(
          field,
          overrides,
          displayConfigFieldLabel(field, section, group.title),
          disabledFieldReasonByKey.get(field.key),
        ),
      ),
      controlField: groupControlField,
      isEnabled,
      isSwitchDisabled:
        !group.controlField ||
        group.controlField.locked === true ||
        (!isEnabled && firstConcreteOption === undefined),
      firstConcreteOption,
      disabledReason: group.disabledReason,
      stackHint: group.stackHint,
      metrics: runtimeDefaultsMetrics(group.fields, overrides),
    } satisfies RuntimeDefaultsFieldGroupPresentation;
  });
  const bodyFields = bodySchemas.map((field) =>
    presentField(
      field,
      overrides,
      displayConfigFieldLabel(field, section),
      disabledFieldReasonByKey.get(field.key),
    ),
  );
  const children = (section.children ?? []).map((child, childIndex) =>
    presentSection({
      section: child,
      sourceSection: sourceSection.children?.find(
        (candidate) => configSectionKey(candidate) === configSectionKey(child),
      ) ?? child,
      allFields,
      disabledFieldReasonByKey,
      overrides,
      showInheritedFields,
      inheritedDisabledReason: disabledReason,
      index: childIndex,
      idPrefix: `${id}-nested-section`,
    }),
  );
  const inheritedHint = showInheritedFields
    ? inheritedHiddenModelFieldHint(
        section,
        allFields,
      )
    : undefined;
  const inheritedField = inheritedHint
    ? {
        label: inheritedHint.label,
        title: inheritedHint.title,
        sourceTitle: inheritedHint.sourceTitle,
        field: presentField(
          inheritedHint.sourceField,
          overrides,
          inheritedHint.label,
          disabledFieldReasonByKey.get(inheritedHint.sourceField.key),
        ),
      }
    : undefined;
  const displayDescription =
    section.title === "Attention Mode" && !controlField
      ? undefined
      : displayConfigSectionDescription(section.title);

  return {
    id,
    key,
    title: section.title,
    displayTitle: displayConfigSectionTitle(section.title),
    displayDescription,
    fields: section.fields.map((field) =>
      presentField(
        field,
        overrides,
        displayConfigFieldLabel(field, section),
        disabledFieldReasonByKey.get(field.key),
      ),
    ),
    bodyFields,
    fieldGroups,
    children,
    controlField,
    disabledReason,
    isDisabled: disabledReason !== undefined,
    directMetrics: runtimeDefaultsMetrics(section.fields, overrides),
    treeMetrics: runtimeDefaultsMetrics(
      configSectionFields(section),
      overrides,
    ),
    stackInheritanceHint: inheritedStackSectionHint(
      sourceSection,
      overrides,
    ),
    inheritedField,
    childKeySignature: children.map((child) => child.key).join("\u0000"),
    groupTitleSignature: (fieldGroups ?? []).map((group) => group.title).join("\u0000"),
    enabledGroupTitleSignature: (fieldGroups ?? [])
      .filter((group) => group.isEnabled)
      .map((group) => group.title)
      .join("\u0000"),
  };
}

function matchesSearchQuery(
  option: RuntimeDefaultsSearchOptionPresentation,
  query: string,
) {
  return configSearchOptionMatchesQuery(option, query);
}

export function presentRuntimeDefaultsSchema({
  sections,
  overrides,
  search,
}: {
  sections: ConfigSection[];
  overrides: OverrideValues;
  search: ConfigSearchState;
}): RuntimeDefaultsSchemaPresentation {
  const allSchemaFields = configSectionsFields(sections);
  const schemaFields = applicableConfigFields(allSchemaFields, overrides);
  const displayFields = schemaFields.filter((field) => shouldPresentConfigField(field, overrides));
  const displaySections = deriveNestedConfigSections(
    groupConfigFieldsBySectionPath(displayFields),
  );
  const schemaMetrics = runtimeDefaultsMetrics(displayFields, overrides);
  const sourceSections = new Map(
    displaySections.map((section) => [configSectionKey(section), section]),
  );
  const disabledFieldReasonByKey = disabledConfigFieldReasons(
    displaySections,
    overrides,
  );
  const selectedFieldKey = search.selectedFieldKey ?? null;
  const isSearchActive = search.query.trim().length > 0 || selectedFieldKey !== null;
  const filteredSections = filterConfigSectionsForSearch(displaySections, {
    query: search.query,
    selectedFieldKey,
  });
  const visibleSections = deriveNestedConfigSections(
    filteredSections,
    displaySections,
  );
  const sectionPresentations = visibleSections.map((section, index) =>
    presentSection({
      section,
      sourceSection: sourceSections.get(configSectionKey(section)) ?? section,
      allFields: schemaFields,
      disabledFieldReasonByKey,
      overrides,
      showInheritedFields: !isSearchActive,
      index,
    }),
  );
  const defaultOpenSectionKeys = isSearchActive
    ? sectionPresentations.map((section) => section.key)
    : sectionPresentations
        .filter((section, index) => index === 0 || section.treeMetrics.overrideCount > 0)
        .map((section) => section.key);
  const searchOptions = flattenConfigSearchOptions(displaySections).map((option) => ({
    sectionTitle: option.sectionTitle,
    rootSectionKey: option.rootSectionKey,
    field: presentField(
      option.field,
      overrides,
      option.label,
      disabledFieldReasonByKey.get(option.key),
    ),
    key: option.key,
    label: option.label,
    configKey: option.configKey,
    flag: option.flag,
    type: option.type,
  }));

  return {
    schemaFields,
    fieldCount: schemaMetrics.fieldCount,
    presetOwnedFieldCount: schemaMetrics.presetCount,
    sections: sectionPresentations,
    defaultOpenSectionKeys,
    searchOpenKey: isSearchActive
      ? `${selectedFieldKey ?? ""}\u0000${search.query.trim()}`
      : "all",
    isSearchActive,
    search: {
      options: searchOptions,
      matchesQuery: matchesSearchQuery,
    },
  };
}
