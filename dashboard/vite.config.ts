import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs separately (uvicorn agenteval.api:app on :8000); proxying it
// keeps the dashboard same-origin, as it is when `agenteval ui` serves both.
// For a separate deployment, set VITE_API_URL at build time instead.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
});
