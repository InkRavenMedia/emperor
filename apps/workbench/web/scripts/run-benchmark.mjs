import { createServer } from "vite";

const benchmark = process.argv[2];
if (!benchmark || !/^benchmark-[a-z0-9-]+\.ts$/.test(benchmark)) {
  throw new Error("Usage: node scripts/run-benchmark.mjs benchmark-<name>.ts");
}

const server = await createServer({
  configFile: "vitest.config.ts",
  server: { middlewareMode: true, watch: null, hmr: false, ws: false },
});
try {
  await server.ssrLoadModule(`/scripts/${benchmark}`);
} finally {
  await server.close();
}
