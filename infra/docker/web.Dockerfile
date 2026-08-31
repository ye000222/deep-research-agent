FROM node:22-alpine

WORKDIR /workspace
RUN corepack enable

COPY package.json pnpm-workspace.yaml ./
COPY apps/web/package.json ./apps/web/package.json
RUN pnpm install --no-frozen-lockfile

COPY apps/web ./apps/web

EXPOSE 5174
CMD ["pnpm", "--filter", "@deep-research/web", "dev", "--host", "0.0.0.0"]
