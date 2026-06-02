/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // brain.js uses native bindings on the server; exclude it from server bundles
  // so it is only loaded in the browser via dynamic import.
  serverExternalPackages: ["brain.js"],
}

export default nextConfig
