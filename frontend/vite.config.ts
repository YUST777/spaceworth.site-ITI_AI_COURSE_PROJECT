import path from "node:path";
import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
  build: {
    target: "esnext",
    cssCodeSplit: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules")) {
            if (id.includes("react") || id.includes("react-dom")) return "vendor-react";
            if (id.includes("konva") || id.includes("react-konva")) return "vendor-konva";
            if (id.includes("framer-motion")) return "vendor-motion";
            if (id.includes("leaflet")) return "vendor-leaflet";
            if (id.includes("lucide-react") || id.includes("driver.js")) return "vendor-ui";
            return "vendor-deps";
          }
        },
      },
    },
  },
});
