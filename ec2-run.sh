#!/usr/bin/env bash
# Run on the EC2 instance to pull the latest image and restart the container.
set -euo pipefail

AWS_REGION="eu-north-1"
AWS_ACCOUNT_ID="180294190448"
ECR_REPO="toto-ai"
IMAGE_TAG="latest"
CONTAINER_NAME="toto-ai"
ENV_FILE="/home/ubuntu/toto-ai.env"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

echo "==> Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "==> Pulling latest image..."
docker pull "$ECR_URI"

echo "==> Stopping old container (if any)..."
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "==> Starting container..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --env-file "$ENV_FILE" \
  "$ECR_URI"

echo "==> Container started. Tailing logs (Ctrl+C to stop following):"
docker logs -f "$CONTAINER_NAME"
