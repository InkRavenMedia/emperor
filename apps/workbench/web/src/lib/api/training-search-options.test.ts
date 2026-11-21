import { describe, expect, it } from "vitest";
import cases from "../../../tests/fixtures/search-options.json";
import { trainingSearchInputSchema } from "@/lib/api/training-jobs";

describe("Workbench API search options", () => {
  it.each(cases)("checks $input", ({ input, accepted, count, field }) => {
    const result = trainingSearchInputSchema.safeParse({
      ...input,
      values: { width: [1, 2] },
    });
    expect(result.success).toBe(accepted);
    if (result.success) {
      expect(result.data.mode === "random" ? result.data.randomSamples : null).toBe(count);
    } else {
      expect(result.error.issues.map((issue) => issue.path)).toContainEqual([field]);
    }
  });
});
