import { describe, expect, it } from "vitest";
import { expectedLinearParameterChannels } from "@/lib/parameter-summary";
import type { GraphNode, InspectResponse } from "@/lib/api/inspection";

function node(
  id: string,
  path: string,
  typeName: string,
  overrides: Partial<GraphNode> = {},
): GraphNode {
  return {
    id,
    label: id,
    typeName,
    path,
    graphRole: overrides.graphRole ?? "architecture",
    parameterCount: overrides.parameterCount ?? 0,
    parameterSizeBytes: overrides.parameterSizeBytes ?? (overrides.parameterCount ?? 0) * 4,
    details: overrides.details ?? {},
    config: overrides.config ?? null,
  };
}

function graph({ withBias = true }: { withBias?: boolean } = {}): InspectResponse {
  const root = node("root", "main_model", "LayerStack");
  const layers = Array.from({ length: 4 }, (_, index) => {
    const wrapper = node(`layer-${index}`, `main_model.layers.${index}`, "Layer");
    const linear = node(
      `linear-${index}`,
      `main_model.layers.${index}.model`,
      "LinearLayer",
      {
        details: withBias
          ? { weightShape: "2 x 2", biasShape: "2" }
          : { weightShape: "2 x 2" },
      },
    );
    return { wrapper, linear };
  });

  return {
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    parameterCount: 0,
    parameterSizeBytes: 0,
    nodes: [root, ...layers.flatMap((layer) => [layer.wrapper, layer.linear])],
    edges: layers.flatMap((layer, index) => [
      { id: `root-layer-${index}`, source: root.id, target: layer.wrapper.id },
      {
        id: `layer-${index}-linear-${index}`,
        source: layer.wrapper.id,
        target: layer.linear.id,
      },
    ]),
  };
}

describe("expected linear parameter channels", () => {
  it("tracks weights and only the biases present in the graph", () => {
    expect(expectedLinearParameterChannels(graph()).map(({channel}) => channel))
      .toEqual(Array.from({length: 4}, () => ["weights", "bias"]).flat());
    expect(expectedLinearParameterChannels(graph({withBias: false})).map(({channel}) => channel))
      .toEqual(["weights", "weights", "weights", "weights"]);
  });
});
