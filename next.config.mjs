/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // Keep tfjs out of server-side bundles — it is dynamically imported client-side only
  serverExternalPackages: ["@tensorflow/tfjs"],
}

export default nextConfig
