import type { NextConfig } from "next";

// /api/v1/* is proxied at runtime by app/api/v1/[...path]/route.ts, which reads LUMINA_API_URL
// per request. A rewrite is deliberately not used: rewrites are resolved when the app is built.
const nextConfig: NextConfig = {};

export default nextConfig;
