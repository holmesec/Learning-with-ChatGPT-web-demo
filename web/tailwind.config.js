/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./chatta/templates/**/*.html"],
  theme: {
    extend: {
      colors: {
        primary: "#0b0219",
        secondary: "#1c1625",
        midnight: "#0c0a09",
      },
      transitionProperty: {
        height: "height",
      },
    },
  },
  plugins: [],
};
