FROM node:18 AS builder

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
RUN npm install react-select
COPY . .
RUN npm run build
