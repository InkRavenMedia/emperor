import { afterEach, expect, it, vi } from "vitest";
import {
  deactivateWorkbenchConnectionRuntime,
  observeWorkbenchAuthMode,
} from "@/lib/api/_connection-runtime";
import { importLogArchive } from "@/lib/api/log-import";

afterEach(() => {
  vi.unstubAllGlobals();
  deactivateWorkbenchConnectionRuntime();
});

it("replays the same multipart bytes for the same archive mutation", async () => {
  observeWorkbenchAuthMode("none");
  const requests: Request[] = [];
  vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
    requests.push(new Request(url, init));
    return Response.json({ extractedFileCount: 1, skippedFileCount: 0, destinationRoot: "/logs" });
  });

  const options = { idempotencyKey: "Archive-RETRY" };
  const archiveName = 'résumé\r\n".zip';
  const archive = new File(["archive data"], archiveName);
  await importLogArchive(archive, options);
  await importLogArchive(archive, options);
  await importLogArchive(new File(["changed data"], archiveName), options);

  const [initial, retry, changed] = requests;
  expect(initial.headers.get("Idempotency-Key")).toBe(options.idempotencyKey);
  expect(initial.headers.get("content-type")).toMatch(/^multipart\/form-data; boundary=/);
  expect(retry.headers.get("content-type")).toBe(initial.headers.get("content-type"));
  const initialBytes = Array.from(new Uint8Array(await initial.clone().arrayBuffer()));
  expect(Array.from(new Uint8Array(await retry.arrayBuffer()))).toEqual(initialBytes);
  expect(Array.from(new Uint8Array(await changed.arrayBuffer()))).not.toEqual(initialBytes);
  expect(new TextDecoder().decode(Uint8Array.from(initialBytes)))
    .toContain('filename="résumé%0D%0A%22.zip"');

  const uploaded = (await initial.formData()).get("archive");
  expect(uploaded).toBeInstanceOf(File);
  if (!(uploaded instanceof File)) throw new Error("Expected a multipart archive file");
  expect(uploaded.name).toBe(archiveName);
  expect(uploaded.type).toBe("application/octet-stream");
  expect(await uploaded.text()).toBe("archive data");
});
