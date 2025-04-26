import json
import base64
import argparse
import boto3
from pathlib import Path


def get_content_type(image_path):
    ext = Path(image_path).suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    else:
        return 'image/png'


def invoke_endpoint(endpoint_name, image_path):
    runtime = boto3.client('sagemaker-runtime')
    
    # Read and encode the image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Prepare the payload
    payload = {
        "image": image_b64,
        "parameters": {
            "input_size": [1024, 1024],
            "output_format": "png"
        }
    }
    
    # Call the endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps(payload)
    )
    
    # Parse the response
    result = json.loads(response['Body'].read().decode())
    
    # Decode and save the mask
    mask_data = base64.b64decode(result['mask'])
    output_path = "mask_output.png"
    with open(output_path, 'wb') as f:
        f.write(mask_data)
    
    print(f"Mask saved to {output_path}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invoke SageMaker endpoint')
    parser.add_argument('--endpoint-name', required=True, help='SageMaker endpoint name')
    parser.add_argument('--image-path', required=True, help='Path to input image')
    parser.add_argument('--region', help='AWS region')
    args = parser.parse_args()
    
    result = invoke_endpoint(args.endpoint_name, args.image_path, args.region) 