#!/usr/bin/env bash
# Run once on the EC2 instance to install Docker and AWS CLI.
set -euo pipefail

AWS_REGION="eu-north-1"
AWS_ACCOUNT_ID="180294190448"

echo "==> Installing Docker..."
sudo apt-get update
sudo apt-get install -y docker.io unzip curl
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo "==> Installing AWS CLI v2..."
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
sudo /tmp/aws/install
rm -rf /tmp/awscliv2.zip /tmp/aws

echo "==> Done. AWS CLI version: $(aws --version)"
echo ""
echo "==> IMPORTANT: Log out and back in (or run 'newgrp docker') for the docker group to take effect."
echo ""
echo "==> Then authenticate to ECR with:"
echo "    aws ecr get-login-password --region ${AWS_REGION} \\"
echo "      | docker login --username AWS --password-stdin \\"
echo "          ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
