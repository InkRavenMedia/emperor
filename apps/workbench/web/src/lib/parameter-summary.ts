import type { GraphNode, InspectResponse } from "@/lib/api/inspection";
import {
  createLinearMonitorTargetResolver,
  type LinearMonitorTargetResolver,
} from "@/lib/graph/monitor-targets";

export type ExpectedParameterChannelName = "weights" | "bias";

export type ExpectedParameterChannel = {
  nodePath: string;
  channel: ExpectedParameterChannelName;
  node: GraphNode;
};

function hasShape(value: unknown) {
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  return (
    Array.isArray(value) &&
    value.every((item) => typeof item === "number" || typeof item === "string")
  );
}

export function expectedLinearParameterChannels(
  graph: InspectResponse | undefined,
  resolver: LinearMonitorTargetResolver = createLinearMonitorTargetResolver(graph),
): ExpectedParameterChannel[] {
  const targets = new Map<string, GraphNode>();
  for (const node of graph?.nodes ?? []) {
    const target = resolver(node);
    if (target) {
      targets.set(target.path, target);
    }
  }

  return [...targets.values()]
    .sort((left, right) => left.path.localeCompare(right.path))
    .flatMap((node) => {
      const channels: ExpectedParameterChannel[] = [
        { nodePath: node.path, channel: "weights", node },
      ];
      if (hasShape(node.details.biasShape)) {
        channels.push({ nodePath: node.path, channel: "bias", node });
      }
      return channels;
    });
}
