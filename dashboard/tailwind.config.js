/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#0a0e17',
        'bg-secondary': '#111827',
        'bg-card': '#1a1f2e',
        'border-dim': '#2a3042',
        'text-primary': '#e5e7eb',
        'text-secondary': '#9ca3af',
        'accent-green': '#10b981',
        'accent-red': '#ef4444',
        'accent-blue': '#3b82f6',
        'accent-yellow': '#f59e0b',
        'accent-purple': '#8b5cf6',
      },
    },
  },
  plugins: [],
}
