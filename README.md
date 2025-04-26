# ISNet Segmentation Model

This repository contains an implementation of the ISNet segmentation model.

## Features

- Implementation of the ISNet segmentation model
- Test suite using pytest
- Deployment via AWS SageMaker
- CI/CD pipeline for automated testing and deployment

## Table of Contents

- [Testing](#testing)
- [Deployment](#deployment)
  - [AWS SageMaker](#aws-sagemaker)
- [CI/CD Pipeline](#cicd-pipeline)

## Testing

Run the tests using pytest:

```bash
python -m pytest src/tests/
```

## Deployment

### AWS SageMaker

1. Build and push the Docker image to Amazon ECR:

```bash
cd deployment/sagemaker
./build_and_push.sh aws-account-id location isnet-segmentation
```

2. Deploy the model to SageMaker:

```bash
python deployment/sagemaker/deploy.py \
  --image-uri aws-account-id.dkr.ecr.location.amazonaws.com/isnet-segmentation:latest \
  --model-data s3://bucket-name/models/isnet/model.tar.gz \
  --endpoint-name isnet-endpoint \
  --instance-type ml.g4dn.xlarge \
  --instance-count 1
```

## CI/CD Pipeline

The CI/CD pipeline is implemented using GitHub Actions. It automates the following steps:

1. Run tests on each pull request
2. Build and push Docker images
3. Deploy to development environment 
4. Deploy to production environment 