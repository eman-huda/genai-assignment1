/** Design tokens: lavender-tinted paper, baby blue panels, lavender accents, slate-navy ink. */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#FBFAFF",
        ink: { DEFAULT: "#283552", soft: "#5B6785", faint: "#8C95AD" },
        sky: { 50: "#F2F8FE", 100: "#E6F1FC", 200: "#DCEBFB", 300: "#BFDCF7", 500: "#6FA8DC" },
        lilac: { 50: "#F6F3FE", 100: "#EEE9FC", 200: "#D9CFF6", 300: "#C2B4EF", 500: "#9584D6", 600: "#7462C0", 700: "#5E4DA6" },
        line: "#E6E2F3",
      },
      fontFamily: { sans: ["Nunito", "ui-rounded", "system-ui", "sans-serif"] },
      borderRadius: { panel: "1.5rem" },
      boxShadow: { soft: "0 1px 2px rgba(40,53,82,.04), 0 8px 24px -12px rgba(116,98,192,.18)" },
    },
  },
  plugins: [],
};
