import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// VITE_API_URL is baked in at build time (Render Static Site env var).
// Local dev default is the FastAPI server on :8000.
export default defineConfig({
  plugins: [react()],
});
