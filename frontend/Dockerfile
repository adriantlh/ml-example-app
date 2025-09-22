# Install dependencies
FROM node:20-alpine AS deps
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci

# Build application
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend .
RUN npm run build-standalone

# Runtime image
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/.next/standalone ./.next/standalone
COPY --from=builder /app/public ./public
# Ensure static assets alongside standalone build
COPY --from=builder /app/.next/static ./.next/standalone/.next/static

EXPOSE 3000

CMD ["node", ".next/standalone/server.js"]
