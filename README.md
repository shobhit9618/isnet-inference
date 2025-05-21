# ISNet Segmentation Model

## Project Overview

ISNet (Image Segmentation Network) is a deep learning model designed for salient object detection. This technique identifies and isolates the most prominent objects in an image. Common use cases for ISNet include background removal in photos, advanced image editing software, and aiding in medical image analysis by highlighting areas of interest. This repository provides a comprehensive implementation of the ISNet model, detailing its architecture and offering tools for model inference, thorough testing, and deployment.

## Table of Contents

- [Running Locally](#running-locally)
- [Model Architecture](#model-architecture)
- [Examples](#examples)
- [Testing](#testing)
- [Deployment](#deployment)
  - [AWS SageMaker](#aws-sagemaker)
- [CI/CD Pipeline](#cicd-pipeline)
- [Contributing](#contributing)
- [License](#license)

## Installation

It is recommended to use a virtual environment to manage dependencies. 

All necessary dependencies are listed in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Features

- Implementation of the ISNet segmentation model
- Test suite using pytest
- Deployment via AWS SageMaker
- CI/CD pipeline for automated testing and deployment

## Running Locally

The `run_local_inference.py` script allows you to run the ISNet segmentation model on your local machine.

**Prerequisites:**

1.  **Pre-trained Model:** You'll need a pre-trained ISNet model file (e.g., `isnet-general-use.pth`). You can typically find these in the original ISNet repository or other public model zoos. This project does not distribute pre-trained models.
2.  **Modify Script Paths:** The script `run_local_inference.py` has hardcoded paths for the model, input image, and output directory. You **must** modify these paths within the script before running it.

    ```python
    # Example modifications in run_local_inference.py
    model_path = "path/to/your/isnet-general-use.pth"  # Replace with the actual path to your model
    image_path = "path/to/your/input_image.jpg"    # Replace with the actual path to your input image
    output_dir = "outputs"                         # Or your desired output directory
    ```

**Execution:**

Once you have the pre-trained model and have updated the paths in the script, you can run the inference:

```bash
python run_local_inference.py
```

**Expected Output:**

The script will process the input image using the ISNet model and save the resulting segmentation mask (as an image file) in the specified `output_dir`.

## Model Architecture

The ISNet model employs a U-Net-like architecture, which is renowned for its effectiveness in image segmentation tasks. A key feature of ISNet is its use of Residual U-blocks (RSU) of varying depths. These blocks are designed to capture contextual information at different scales, allowing for robust feature extraction.

The implementation in this repository, found in `src/models/isnet.py`, defines several RSU blocks:
- `RSU7`
- `RSU6`
- `RSU5`
- `RSU4`
- `RSU4F` (a dilated version of RSU4)

These RSU blocks are integrated into an encoder-decoder structure within the `ISNetDIS` module. The encoder part progressively downsamples the input image while extracting features using the RSU blocks. The decoder part then upsamples these features, combining them with skip connections from the encoder, to reconstruct the final segmentation map.

For a detailed understanding of the architecture, please refer to the original paper: [Highly Accurate Dichotomous Image Segmentation](https://arxiv.org/abs/2108.12322) and the authors' repository: [https://github.com/xuebinqin/DIS](https://github.com/xuebinqin/DIS).

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
