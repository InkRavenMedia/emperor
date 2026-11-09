import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  reactCompiler: true,
  typedRoutes: true,
  turbopack: { root: __dirname },
  experimental: {
    // Cached loader paths can survive a workspace-root change and break HMR.
    turbopackFileSystemCacheForDev: false,
  },
};

export default nextConfig;
