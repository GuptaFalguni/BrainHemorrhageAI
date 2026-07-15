import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  async redirects() {
    return [
      { source: "/analyze", destination: "/", permanent: true },
      { source: "/runs", destination: "/", permanent: true },
      { source: "/models", destination: "/", permanent: true },
      { source: "/compare", destination: "/", permanent: true },
      { source: "/experiments", destination: "/", permanent: true },
      { source: "/docs", destination: "/", permanent: true },
      { source: "/about", destination: "/", permanent: true },
    ];
  },
};

export default nextConfig;
