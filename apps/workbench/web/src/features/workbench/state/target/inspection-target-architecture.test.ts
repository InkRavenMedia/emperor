import ts from "typescript";
import { describe, expect, it } from "vitest";
import {
  callsNamed,
  declaredUseStateSetters,
  readTypeScriptSource,
  typeLiteralPropertyNames,
} from "@/test-utils/typescript-ast";

const orchestration = readTypeScriptSource(
  "src/features/workbench/state/target/use-model-package-inspection-state.ts",
);
const lifecycle = readTypeScriptSource(
  "src/features/workbench/state/target/_inspection-target-state.ts",
);

function isIdentifierArgument(
  argument: ts.Expression | undefined,
  name: string,
) {
  return argument !== undefined && ts.isIdentifier(argument) && argument.text === name;
}

const targetSetters = [
  "setSelectedModel",
  "setSelectedModelType",
  "setSelectedPreset",
  "setSelectedExperimentTask",
  "setSelectedDatasets",
  "setPresetOverrides",
  "setInspectionTransition",
];

function misplacedTargetSetters(source: ts.SourceFile) {
  return declaredUseStateSetters(source).filter((setter) => targetSetters.includes(setter));
}

describe("Inspection target architecture", () => {
  it.each(targetSetters)("recognizes a single misplaced %s setter", (setter) => {
    const source = ts.createSourceFile(
      "fixture.ts",
      `const [value, ${setter}] = useState(null);`,
      ts.ScriptTarget.Latest,
      true,
    );
    expect(misplacedTargetSetters(source)).toEqual([setter]);
  });

  it("keeps every complete-target field and semantic revision in the lifecycle Module", () => {
    expect(misplacedTargetSetters(orchestration)).toEqual([]);
    expect(
      typeLiteralPropertyNames(lifecycle, "InspectionTargetLifecycleState"),
    ).toEqual(
      expect.arrayContaining([
        "modelPackage",
        "selectedPreset",
        "target",
        "experimentTask",
        "datasets",
        "runtimeDefaults",
        "restoration",
        "transition",
      ]),
    );
  });

  it("derives Inspection request identity once from the complete lifecycle", () => {
    expect(callsNamed(orchestration, "resolveInspectionTarget")).toHaveLength(1);
    expect(
      callsNamed(orchestration, "requestPreview").some((call) =>
        isIdentifierArgument(call.arguments[0], "currentInspectionRequest"),
      ),
    ).toBe(true);
    expect(
      callsNamed(orchestration, "issueInspectionPreview").some(
        (call) =>
          isIdentifierArgument(
            call.arguments[0],
            "currentInspectionRequest",
          ) &&
          call.arguments[1] !== undefined &&
          ts.isStringLiteralLike(call.arguments[1]) &&
          call.arguments[1].text === "refresh",
      ),
    ).toBe(true);
  });
});
