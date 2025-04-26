#!/bin/bash
# Build and push Docker image for SageMaker

# Usage: ./build_and_push.sh [aws_account_id] [region] [repository_name] [tag]

set -e

# Default values
AWS_ACCOUNT_ID=${1:-$(aws sts get-caller-identity --query Account --output text)}
REGION=${2:-$(aws configure get region)}
REPOSITORY_NAME=${3:-"isnet-segmentation"}
TAG=${4:-"latest"}

# Set up the full image name
FULLNAME="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:${TAG}"

# Log in to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Check if the repository exists, and create it if it doesn't
echo "Checking if repository exists..."
if ! aws ecr describe-repositories --repository-names ${REPOSITORY_NAME} --region ${REGION} 2>&1 > /dev/null; then
    echo "Creating repository ${REPOSITORY_NAME}..."
    aws ecr create-repository --repository-name ${REPOSITORY_NAME} --region ${REGION}
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t ${REPOSITORY_NAME}:${TAG} -f deployment/sagemaker/Dockerfile .

# Tag the image for ECR
echo "Tagging image for ECR..."
docker tag ${REPOSITORY_NAME}:${TAG} ${FULLNAME}

# Push the image to ECR
echo "Pushing image to ECR..."
docker push ${FULLNAME}

echo "Done! Image URI: ${FULLNAME}" 