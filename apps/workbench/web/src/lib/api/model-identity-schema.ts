import { z } from "zod";
import { MODEL_IDENTITY_SEGMENT_PATTERN } from "@/lib/api/model-identity-pattern";

export const modelIdentitySegmentSchema = z
  .string()
  .regex(MODEL_IDENTITY_SEGMENT_PATTERN);

export const modelIdentitySchema = z.object({
  modelType: modelIdentitySegmentSchema,
  model: modelIdentitySegmentSchema,
});

export type ModelIdentity = z.infer<typeof modelIdentitySchema>;
