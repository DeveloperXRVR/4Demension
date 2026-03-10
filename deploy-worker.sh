#!/bin/bash

# 4Demension Worker Deployment Script
# Rebuilds and deploys updated handler to RunPod

set -e

echo "🚀 4Demension Worker Deployment Script"
echo "======================================"

# Configuration
DOCKER_IMAGE="4demension-worker:latest"
DOCKER_REGISTRY="your-dockerhub-username"  # Change to your Docker Hub username
IMAGE_TAG="${DOCKER_REGISTRY}/4demension-worker:latest"

echo "📋 Configuration:"
echo "  Docker Image: $DOCKER_IMAGE"
echo "  Registry: $IMAGE_TAG"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop first."
    echo "   Visit: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker found"

# Step 1: Build the Docker image
echo ""
echo "🔨 Step 1: Building Docker image..."
echo "  This includes the updated handler with mesh generation and format conversion"
docker build -t $DOCKER_IMAGE .
echo "✅ Docker image built successfully"

# Step 2: Tag for registry
echo ""
echo "🏷️  Step 2: Tagging image for registry..."
docker tag $DOCKER_IMAGE $IMAGE_TAG
echo "✅ Image tagged as: $IMAGE_TAG"

# Step 3: Push to registry
echo ""
echo "📤 Step 3: Pushing to Docker registry..."
echo "  Make sure you're logged in: docker login"
docker push $IMAGE_TAG
echo "✅ Image pushed to registry"

# Step 4: Instructions for RunPod update
echo ""
echo "🔄 Step 4: RunPod Update Required"
echo "  ================================="
echo "  The image has been pushed, but you need to manually update RunPod:"
echo ""
echo "  1. Go to https://www.runpod.io/console/serverless"
echo "  2. Find your endpoint (ID: ika5cdpb25jjr5)"
echo "  3. Click 'Edit' or 'Update'"
echo "  4. Update Docker image to: $IMAGE_TAG"
echo "  5. Save changes"
echo "  6. Wait for workers to restart"
echo ""
echo "  🎯 Your new features will be available after restart:"
echo "     • Increased point cloud density (1x-3x)"
echo "     • Mesh generation (Poisson/Ball Pivot/Alpha)"
echo "     • Multiple format export (OBJ, GLB, DAE, STL, PLY)"
echo "     • Texture mapping"

# Step 5: Git commit (optional)
echo ""
echo "📝 Step 5: Git Commit (Optional)"
echo "  ==============================="
echo "  Don't forget to commit your changes to GitHub:"
echo ""
echo "  cd /Users/macos/Desktop/Developer/4Demension"
echo "  git add ."
echo "  git commit -m \"Enhanced 4Demension: mesh generation + multi-format export\""
echo "  git push origin main"
echo ""

echo "🎉 Deployment script completed!"
echo "   Next: Update RunPod endpoint manually"
