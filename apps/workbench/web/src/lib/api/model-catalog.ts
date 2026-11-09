import { array, object, regex, string } from "zod/v4-mini";

import { requestJson } from "@/lib/api/client";
import { MODEL_IDENTITY_SEGMENT_PATTERN } from "@/lib/api/model-identity-pattern";

const modelIdentitySegmentSchema = string().check(regex(MODEL_IDENTITY_SEGMENT_PATTERN));
const modelIdentitySchema = object({
  modelType: modelIdentitySegmentSchema,
  model: modelIdentitySegmentSchema,
});
const modelsSchema = object({ models: array(modelIdentitySchema) });

export type { ModelIdentity } from "@/lib/api/model-identity-schema";

type ApiRequestOptions = {
  signal?: AbortSignal;
};

export function fetchModels(options: ApiRequestOptions = {}) {
  return requestJson("/models", modelsSchema, options, {
    authenticationProbe: true,
  });
}
