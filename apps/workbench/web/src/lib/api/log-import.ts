import { z } from "zod";

import {
  requestMultipartJson,
  type MutationRequestOptions,
} from "@/lib/api/client";

type ApiRequestOptions = MutationRequestOptions & {
  signal?: AbortSignal;
};

export const logArchiveImportSchema = z.object({
  extractedFileCount: z.number().int().nonnegative(),
  skippedFileCount: z.number().int().nonnegative(),
  destinationRoot: z.string(),
});

export type LogArchiveImportResponse = z.infer<typeof logArchiveImportSchema>;

export function importLogArchive(file: File, options: ApiRequestOptions) {
  // The mutation journal compares wire bytes. Keep the same multipart boundary
  // on retries and compose the File without reading it into JavaScript memory.
  const keyToken = options.idempotencyKey
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "")
    .slice(0, 48);
  const boundary = `workbench-${keyToken}`;
  const filename = file.name
    .replaceAll("\r", "%0D")
    .replaceAll("\n", "%0A")
    .replaceAll('"', "%22");
  const body = new Blob(
    [
      `--${boundary}\r\nContent-Disposition: form-data; name="archive"; filename="${filename}"\r\n`,
      `Content-Type: ${file.type || "application/octet-stream"}\r\n\r\n`,
      file,
      `\r\n--${boundary}--\r\n`,
    ],
    { type: `multipart/form-data; boundary=${boundary}` },
  );
  return requestMultipartJson(
    "/logs/import",
    logArchiveImportSchema,
    body,
    {
      method: "POST",
      signal: options.signal,
    },
    { mutation: options },
  );
}
