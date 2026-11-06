import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // GitHub Pages serves the app from /<repo>/, local dev stays at /.
  base: process.env.VITE_BASE_PATH || '/',
  server: {
    port: 5173,
    open: false, // Set to false for headless/dev stability
    host: true,
    // Demo tunnel (ngrok) is a different hostname; allow it so the jury can open the link.
    allowedHosts: true,
    proxy: { '/api': 'http://127.0.0.1:8000', '/health': 'http://127.0.0.1:8000' },
  },
  preview: {
    port: 4173,
    host: true,
    allowedHosts: true,
    proxy: { '/api': 'http://127.0.0.1:8000', '/health': 'http://127.0.0.1:8000' },
  },
});
