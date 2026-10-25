import type { NextConfig } from "next";

/** Address of the Lumina API server that /api/v1/* is proxied to. */
const apiUrl = (process.env.LUMINA_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${apiUrl}/api/v1/:path*`,
      },
    ];
  },
  experimental: {
    // Next buffers proxied request bodies (default 10 MB) and silently truncates
    // anything larger. Uploads may be up to 20 MB plus multipart framing.
    proxyClientMaxBodySize: "21mb",
    // Proxy timeout in ms (Next default is 30 s); CPU inference can take longer.
    proxyTimeout: 120_000,
  },
};

export default nextConfig;
