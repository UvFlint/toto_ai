#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="eu-north-1"
AWS_ACCOUNT_ID="180294190448"
ECR_REPO="toto-ai"
IMAGE_TAG="latest"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo "==> Logging into ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "==> Creating ECR repo (idempotent)..."
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" \
  || aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION"

echo "==> Building image..."
docker build -t "${ECR_REPO}:${IMAGE_TAG}" .

echo "==> Tagging..."
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"

echo "==> Pushing to ECR..."
docker push "${ECR_URI}:${IMAGE_TAG}"

echo "==> Done. Image available at: ${ECR_URI}:${IMAGE_TAG}"
