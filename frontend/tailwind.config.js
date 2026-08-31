/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#070b14",
          900: "#0b1120",
          850: "#0f172a",
          800: "#131c31",
          700: "#1e293b",
          600: "#334155",
        },
        accent: {
          DEFAULT: "#38bdf8",
          soft: "#7dd3fc",
          deep: "#0284c7",
        },
        alert: "#fb7185",
        clear: "#34d399",
        warn: "#fbbf24",
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgb(255 255 255 / 0.04) inset, 0 12px 32px -16px rgb(0 0 0 / 0.8)",
      },
      keyframes: {
        sweep: { "0%": { transform: "translateY(-100%)" }, "100%": { transform: "translateY(400%)" } },
        fadeIn: { from: { opacity: "0", transform: "translateY(4px)" }, to: { opacity: "1", transform: "none" } },
      },
      animation: {
        sweep: "sweep 1.6s ease-in-out infinite",
        fadeIn: "fadeIn 0.25s ease-out both",
      },
    },
  },
  plugins: [],
};
