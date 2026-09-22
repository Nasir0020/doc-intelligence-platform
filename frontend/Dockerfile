# syntax=docker/dockerfile:1

# Multi-stage build: the "builder" stage installs dependencies and runs
# `next build`, producing optimized static/server output. The final
# stage copies ONLY that build output into a fresh, smaller image —
# node_modules' dev dependencies (TypeScript compiler, Tailwind's CLI,
# etc.) never end up in the image that actually runs in production,
# which meaningfully shrinks the final image size and reduces attack
# surface (fewer packages present = fewer potential vulnerabilities to
# patch in the deployed artifact).
FROM node:22-slim AS builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM node:22-slim AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000
CMD ["npm", "start"]
