/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    unoptimized: true // Raspberry Pi'de daha iyi performans için
  },
  output: 'standalone' // Daha iyi production build için
}

module.exports = nextConfig
