import { describe, expect, it } from "vitest";
import type { ConfigField } from "@/lib/api/models";
import { type ConfigSection } from "@/lib/config";
import { presentRuntimeDefaultsSchema } from "@/features/workbench/state/runtime-defaults/runtime-defaults-presentation";

function field(
  overrides: Partial<ConfigField> &
    Pick<ConfigField, "key" | "section" | "sectionPath">,
): ConfigField {
  return {
    key: overrides.key,
    configKey: overrides.configKey ?? overrides.key.toUpperCase(),
    flag: overrides.flag ?? `--${overrides.key.replace(/_/g, "-")}`,
    label: overrides.label ?? overrides.key,
    section: overrides.section,
    sectionPath: overrides.sectionPath,
    sectionIdPath: overrides.sectionIdPath,
    sectionControlFieldKey: overrides.sectionControlFieldKey,
    description: overrides.description,
    type: overrides.type ?? "int",
    legacyJson: overrides.legacyJson,
    default: Object.prototype.hasOwnProperty.call(overrides, "default")
      ? (overrides.default ?? null)
      : 0,
    nullable: overrides.nullable ?? false,
    choices: overrides.choices ?? [],
    applicableWhen: overrides.applicableWhen ?? [],
    locked: overrides.locked ?? false,
    lockedValue: overrides.lockedValue,
    lockedReason: overrides.lockedReason,
  };
}

function schemaSections(): ConfigSection[] {
  const globalHidden = field({
    key: "hidden_dim",
    label: "Hidden Dim",
    section: "Global",
    sectionPath: ["Global"],
    default: 64,
    locked: true,
    lockedValue: 96,
    lockedReason: "Selected preset owns this value.",
  });
  const layerStackDepth = field({
    key: "stack_depth",
    label: "Stack Depth",
    section: "Layer Stack Options",
    sectionPath: ["Layer Stack Options"],
    default: 2,
  });
  const gateFlag = field({
    key: "stack_gate_flag",
    label: "Gate Flag",
    section: "Gate Options",
    sectionPath: ["Gate Options"],
    type: "bool",
    default: false,
    choices: [true, false],
  });
  const gateScale = field({
    key: "gate_scale",
    label: "Gate Scale",
    section: "Gate Options",
    sectionPath: ["Gate Options"],
    type: "float",
    default: 1,
  });
  const gateStackFlag = field({
    key: "gate_stack_independent_flag",
    label: "Gate Stack Independent Flag",
    section: "Gate Stack Options",
    sectionPath: ["Gate Options", "Gate Stack Options"],
    type: "bool",
    default: false,
    choices: [true, false],
  });
  const gateStackDepth = field({
    key: "gate_stack_depth",
    label: "Gate Stack Depth",
    section: "Gate Stack Options",
    sectionPath: ["Gate Options", "Gate Stack Options"],
    default: 2,
  });
  const inputWeightOption = field({
    key: "input_layer_weight_option",
    label: "Input Layer Weight Option",
    section: "Input Boundary Model Options",
    sectionPath: ["Input Boundary Model Options"],
    type: "class",
    default: null,
    nullable: true,
    choices: ["DynamicWeightConfig"],
  });
  const inputWeightRank = field({
    key: "input_layer_weight_rank",
    label: "Input Layer Weight Rank",
    section: "Input Boundary Model Options",
    sectionPath: ["Input Boundary Model Options"],
    default: 4,
  });

  return [
    { title: "Global", fields: [globalHidden] },
    { title: "Layer Stack Options", fields: [layerStackDepth] },
    { title: "Gate Options", fields: [gateFlag, gateScale] },
    {
      title: "Gate Stack Options",
      fields: [gateStackFlag, gateStackDepth],
    },
    {
      title: "Input Boundary Model Options",
      fields: [inputWeightOption, inputWeightRank],
    },
  ];
}

describe("Runtime Defaults schema presentation", () => {
  it("hides empty legacy model objects while preserving their schema and concrete values", () => {
    const legacyField = (
      key: string,
      values: Partial<ConfigField> = {},
    ) => field({
      key,
      section: "Model Options",
      sectionPath: ["Model Options"],
      type: "json",
      legacyJson: true,
      default: null,
      nullable: true,
      ...values,
    });
    const concrete = '{"type":"LinearLayerConfig","bias_flag":true}';
    const fields = [
      legacyField("ATTN_SOURCE_MEMORY_MODEL_CONFIG"),
      legacyField("ENCODER_ATTN_SOURCE_MEMORY_MODEL_CONFIG", { default: "null" }),
      legacyField("SEQUENCE_EXPANSION_MODEL_CONFIG", { default: concrete }),
      legacyField("ENCODER_SEQUENCE_EXPANSION_MODEL_CONFIG", {
        locked: true,
        lockedValue: concrete,
      }),
      legacyField("DECODER_SEQUENCE_EXPANSION_MODEL_CONFIG"),
      legacyField("ATTN_SOURCE_MEMORY_GROUPING_MODEL_CONFIG", { legacyJson: undefined }),
    ];
    const input = {
      sections: [{ title: "Model Options", fields }],
      search: { query: "" },
    };
    const presentation = presentRuntimeDefaultsSchema({
      ...input,
      overrides: { decoder_sequence_expansion_model_config: concrete },
    });

    const visibleFields = fields.slice(2).map(({ key }) => key);
    expect(presentation.schemaFields.map(({ key }) => key)).toEqual(fields.map(({ key }) => key));
    expect(presentation.fieldCount).toBe(visibleFields.length);
    expect(presentation.sections[0]?.bodyFields.map(({ key }) => key)).toEqual(visibleFields);
    expect(presentation.search.options.map(({ key }) => key)).toEqual(visibleFields);
    expect(presentation.search.options.find(({ key }) => key === "DECODER_SEQUENCE_EXPANSION_MODEL_CONFIG")?.field)
      .toMatchObject({ value: concrete, isModified: true });

    const reset = presentRuntimeDefaultsSchema({ ...input, overrides: {} });
    expect(reset.schemaFields).toHaveLength(fields.length);
    expect(reset.search.options.map(({ key }) => key))
      .not.toContain("DECODER_SEQUENCE_EXPANSION_MODEL_CONFIG");
  });

  it("preserves controls, inheritance and search identity after heading renames", () => {
    const metadata = {
      Global: { id: "hidden_dim", control: null },
      "Layer Stack Options": { id: "stack_num_layers", control: null },
      "Gate Options": { id: "stack_gate_flag", control: "stack_gate_flag" },
      "Gate Stack Options": {
        id: "gate_stack_independent_flag",
        control: "gate_stack_independent_flag",
      },
      "Input Boundary Model Options": { id: "input_layer_weight_option", control: null },
    };
    const sections = schemaSections().map((section) => ({
      ...section,
      fields: section.fields.map((field) => ({
        ...field,
        section: `Renamed ${field.section}`,
        sectionPath: field.sectionPath.map((title) => `Renamed ${title}`),
        sectionIdPath: field.sectionPath.map((title) => metadata[title as keyof typeof metadata].id),
        sectionControlFieldKey: metadata[field.section as keyof typeof metadata].control,
      })),
    }));
    const presentation = presentRuntimeDefaultsSchema({
      sections,
      overrides: { stack_gate_flag: "true" },
      search: { query: "" },
    });

    expect(presentation.sections[1]).toMatchObject({
      title: "Renamed Layer Stack Options",
      inheritedField: { sourceTitle: "Renamed Global", field: { key: "hidden_dim" } },
    });
    expect(presentation.sections[2]).toMatchObject({
      key: "stack_gate_flag",
      controlField: { key: "stack_gate_flag", label: "Enabled" },
      isDisabled: false,
      children: [{
        key: "stack_gate_flag/gate_stack_independent_flag",
        isDisabled: true,
        stackInheritanceHint: { label: "Inherits Layer Stack Submodule" },
      }],
    });
    expect(presentation.sections[3]?.fieldGroups?.[0]).toMatchObject({
      isEnabled: false,
      controlField: { key: "input_layer_weight_option" },
    });
    const filtered = presentRuntimeDefaultsSchema({
      sections,
      overrides: {},
      search: { query: "", selectedFieldKey: "gate_stack_depth" },
    });
    expect(filtered.sections[0]).toMatchObject({
      id: presentation.sections[2]?.id,
      controlField: { key: "stack_gate_flag" },
      children: [{ controlField: { key: "gate_stack_independent_flag" } }],
    });
    expect(filtered.search.options.find(({ key }) => key === "gate_stack_depth"))
      .toMatchObject({ rootSectionKey: "stack_gate_flag" });
  });

  it("keeps repeated section titles independent in search and control lookup", () => {
    const sections = [{
      title: "Settings",
      fields: [
        field({
          key: "stack_gate_flag",
          section: "Details",
          sectionPath: ["Settings", "Details"],
          sectionIdPath: ["gate", "gate_details"],
          sectionControlFieldKey: "stack_gate_flag",
          type: "bool",
          default: false,
        }),
        field({
          key: "memory_slots",
          section: "Details",
          sectionPath: ["Settings", "Details"],
          sectionIdPath: ["memory", "memory_details"],
          sectionControlFieldKey: null,
        }),
      ],
    }];
    const presentation = presentRuntimeDefaultsSchema({
      sections,
      overrides: {},
      search: { query: "" },
    });

    expect(presentation.sections.map(({ key, title }) => ({ key, title }))).toEqual([
      { key: "gate", title: "Settings" },
      { key: "memory", title: "Settings" },
    ]);
    expect(presentation.sections[0]?.children[0]).toMatchObject({ isDisabled: true });
    expect(presentation.sections[1]?.children[0]).toMatchObject({ isDisabled: false });
    expect(presentation.search.options.map(({ rootSectionKey }) => rootSectionKey))
      .toEqual(["gate", "memory"]);
  });

  it("removes inapplicable fields, empty sections, search entries, and counts", () => {
    const selector = field({
      key: "recurrent_composition_option",
      section: "Recurrent Layer Options",
      sectionPath: ["Recurrent Layer Options"],
      type: "class",
      default: "RecurrentLayerConfig",
      choices: [
        "RecurrentLayerConfig",
        "TinyRecursiveModelRecurrentConfig",
      ],
    });
    const shared = field({
      key: "recurrent_no_gradient_transition_count",
      section: "Recurrent Layer Options",
      sectionPath: ["Recurrent Layer Options"],
      default: null,
      nullable: true,
    });
    const standard = field({
      key: "recurrent_max_steps",
      section: "Standard Recurrent Options",
      sectionPath: ["Standard Recurrent Options"],
      applicableWhen: [
        {
          key: "RECURRENT_COMPOSITION_OPTION",
          values: ["RecurrentLayerConfig"],
        },
      ],
    });
    const trm = field({
      key: "recurrent_answer_update_count",
      section: "TRM Recurrent Options",
      sectionPath: ["TRM Recurrent Options"],
      applicableWhen: [
        {
          key: "RECURRENT_COMPOSITION_OPTION",
          values: ["TinyRecursiveModelRecurrentConfig"],
        },
      ],
    });
    const sections = [
      { title: "Recurrent Layer Options", fields: [selector, shared] },
      { title: "Standard Recurrent Options", fields: [standard] },
      { title: "TRM Recurrent Options", fields: [trm] },
    ];

    const standardPresentation = presentRuntimeDefaultsSchema({
      sections,
      overrides: {},
      search: { query: "" },
    });
    expect(standardPresentation.fieldCount).toBe(3);
    expect(standardPresentation.sections.map((item) => item.title)).toEqual([
      "Recurrent Layer Options",
      "Standard Recurrent Options",
    ]);
    expect(standardPresentation.search.options.map((item) => item.key))
      .not.toContain("recurrent_answer_update_count");

    const trmPresentation = presentRuntimeDefaultsSchema({
      sections,
      overrides: {
        recurrent_composition_option: "TinyRecursiveModelRecurrentConfig",
        recurrent_max_steps: "9",
        recurrent_answer_update_count: "3",
      },
      search: { query: "" },
    });
    expect(trmPresentation.fieldCount).toBe(3);
    expect(trmPresentation.sections.map((item) => item.title)).toEqual([
      "Recurrent Layer Options",
      "TRM Recurrent Options",
    ]);
    expect(
      trmPresentation.sections.find(
        (item) => item.title === "TRM Recurrent Options",
      )?.treeMetrics.overrideCount,
    ).toBe(1);
    expect(trmPresentation.search.options.map((item) => item.key))
      .not.toContain("recurrent_max_steps");
  });

  it("assembles navigation counts, nested disablement, inheritance, and field groups", () => {
    const presentation = presentRuntimeDefaultsSchema({
      sections: schemaSections(),
      overrides: { stack_depth: "4" },
      search: { query: "" },
    });

    expect(presentation.fieldCount).toBe(8);
    expect(presentation.presetOwnedFieldCount).toBe(1);
    expect(presentation.sections.map((section) => section.title)).toEqual([
      "Global",
      "Layer Stack Options",
      "Gate Options",
      "Input Boundary Model Options",
    ]);
    expect(presentation.defaultOpenSectionKeys).toEqual([
      "Global",
      "Layer Stack Options",
    ]);

    const global = presentation.sections[0]!;
    expect(global.treeMetrics).toEqual({
      fieldCount: 1,
      overrideCount: 0,
      presetCount: 1,
      state: "preset",
    });

    const layerStack = presentation.sections[1]!;
    expect(layerStack.displayTitle).toBe("Layer Hidden Stack Options");
    expect(layerStack.directMetrics.state).toBe("override");
    expect(layerStack.inheritedField).toMatchObject({
      label: "Hidden Dim",
      sourceTitle: "Global",
      field: { value: "96", isPresetOwned: true },
    });

    const gate = presentation.sections[2]!;
    expect(gate.isDisabled).toBe(true);
    expect(gate.controlField).toMatchObject({
      key: "stack_gate_flag",
      label: "Enabled",
      isEnabledValue: false,
      disabledReason: undefined,
    });
    expect(gate.treeMetrics.fieldCount).toBe(4);
    expect(gate.bodyFields.find((item) => item.key === "gate_scale")?.disabledReason)
      .toBe("Enable Gate Flag before editing Gate Options.");
    expect(gate.children).toHaveLength(1);
    expect(gate.children[0]).toMatchObject({
      title: "Gate Stack Options",
      isDisabled: true,
      disabledReason: "Enable Gate Flag before editing Gate Options.",
    });

    const inputBoundary = presentation.sections[3]!;
    const weightGroup = inputBoundary.fieldGroups?.[0];
    expect(weightGroup).toMatchObject({
      title: "Weight",
      isEnabled: false,
      isSwitchDisabled: false,
      firstConcreteOption: "DynamicWeightConfig",
      metrics: {
        fieldCount: 2,
        overrideCount: 0,
        presetCount: 0,
        state: "default",
      },
    });
    expect(
      weightGroup?.fields.find((item) => item.key === "input_layer_weight_rank")
        ?.disabledReason,
    ).toBe("Select Input Layer Weight Option before editing weight boundary settings.");
  });

  it("keeps the matching leaf and required controls in one searchable tree", () => {
    const presentation = presentRuntimeDefaultsSchema({
      sections: schemaSections(),
      overrides: {},
      search: { query: "", selectedFieldKey: "gate_stack_depth" },
    });

    expect(presentation.isSearchActive).toBe(true);
    expect(presentation.searchOpenKey).toBe("gate_stack_depth\u0000");
    expect(presentation.defaultOpenSectionKeys).toEqual(["Gate Options"]);
    expect(presentation.sections).toHaveLength(1);

    const gate = presentation.sections[0]!;
    expect(gate.fields.map((item) => item.key)).toEqual(["stack_gate_flag"]);
    expect(gate.children[0]?.fields.map((item) => item.key)).toEqual([
      "gate_stack_independent_flag",
      "gate_stack_depth",
    ]);
    expect(gate.children[0]?.inheritedField).toBeUndefined();

    const searchOption = presentation.search.options.find(
      (option) => option.key === "gate_stack_depth",
    );
    expect(searchOption).toMatchObject({
      sectionTitle: "Gate Stack Options",
      rootSectionKey: "Gate Options",
      field: {
        key: "gate_stack_depth",
        value: "2",
        disabledReason: "Enable Gate Flag before editing Gate Options.",
      },
    });
    expect(
      searchOption && presentation.search.matchesQuery(searchOption, "STACK DEPTH"),
    ).toBe(true);
    expect(
      searchOption && presentation.search.matchesQuery(searchOption, "not present"),
    ).toBe(false);
  });

  it("re-derives controlled and inherited states from current overrides", () => {
    const presentation = presentRuntimeDefaultsSchema({
      sections: schemaSections(),
      overrides: {
        stack_gate_flag: "true",
        gate_stack_independent_flag: "false",
      },
      search: { query: "" },
    });

    const gate = presentation.sections.find(
      (section) => section.title === "Gate Options",
    )!;
    const gateStack = gate.children[0]!;
    expect(gate.isDisabled).toBe(false);
    expect(gate.treeMetrics).toMatchObject({
      overrideCount: 2,
      state: "override",
    });
    expect(gateStack).toMatchObject({
      isDisabled: true,
      disabledReason:
        "Enable Gate Stack Independent Flag before editing Gate Stack Options.",
      stackInheritanceHint: {
        isCustom: false,
        sourceTitle: "Layer Stack Submodule Options",
      },
    });
    expect(presentation.defaultOpenSectionKeys).toEqual(["Global", "Gate Options"]);
  });
});
