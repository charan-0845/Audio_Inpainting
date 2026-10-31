import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // DPAI Design System - Light theme (default)
        bg: '#F5F5F0',           // very light grey page background
        surface: '#FFFFFF',       // white cards
        raised: '#FAFAF8',        // slightly elevated
        border: 'rgba(0, 0, 0, 0.08)',
        'border-strong': 'rgba(0, 0, 0, 0.14)',
        fg: '#1D1D1B',            // charcoal primary text
        'fg-muted': '#6B6B66',    // muted secondary text
        primary: '#6D28D9',       // refined purple primary
        'primary-hover': '#5B21B6',
        'primary-foreground': '#FFFFFF',
        missing: '#0D9488',       // teal for auto-detected missing regions
        recon: '#EA580C',         // orange for reconstructed regions
        manual: '#6D28D9',        // purple for manual missing regions
        danger: '#DC2626',
        ring: '#6D28D9',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['IBM Plex Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      borderRadius: {
        'xs': '4px',
        'sm': '8px',
        'DEFAULT': '10px',
        'md': '12px',
        'lg': '16px',
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'card': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
        'elevated': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
      },
      transitionDuration: {
        'fast': '140ms',
      },
      transitionTimingFunction: {
        'ease-out': 'ease-out',
      },
    },
  },
  plugins: [],
}

export default config