import {defineConfig} from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    // 5173 可能被其他本地项目（如 bm-web 容器）占用，固定使用 5174
    port: 5174,
    strictPort: true,
  },
});
