#!/bin/bash
# Script to build and launch the CapitolWatch application using Docker.

echo "Building Docker images and starting containers..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running. Please start Docker and try again."
  exit 1
fi

# Build and run with docker-compose
docker compose up --build -d

echo ""
echo "The application is available at http://localhost:8501"
echo "To stop the application, run: docker compose down"