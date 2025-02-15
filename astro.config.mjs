import react from "@astrojs/react";
// @ts-check
import { defineConfig } from "astro/config";

export default defineConfig({
  output: "static",
  srcDir: "./astro",
  integrations: [react()],
});
