import type { NextConfig } from "next";

// NEXT_OUTPUT env var controls build mode:
//   not set        → dev mode (default)
//   "standalone"   → Docker deployment
//   "export"       → static HTML (set NEXT_BASE_PATH too)

const outputMode = process.env.NEXT_OUTPUT as "standalone" | "export" | undefined;
const basePath = process.env.NEXT_BASE_PATH || "";

const nextConfig: NextConfig = {
  ...(outputMode ? { output: outputMode } : {}),
  ...(basePath ? { basePath } : {}),
  reactCompiler: true,
  images: { unoptimized: true },
  trailingSlash: true,
  allowedDevOrigins: ["http://127.0.0.1:*", "http://localhost:*"],
  async headers() {
    return [
      {
        source: "/viewer/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
    ];
  },
};

export default nextConfig;
