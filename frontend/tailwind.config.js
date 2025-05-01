/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#00A67E',
          light: '#E6F6F2',
        },
        secondary: {
          DEFAULT: '#1E293B',
        },
        background: {
          DEFAULT: '#F8FAFC',
          dark: '#1E293B',
        }
      },
    },
  },
  plugins: [],
} 