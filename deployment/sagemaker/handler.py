"""
AWS SageMaker handler for the model
"""
import os
import io
import json
import base64
import logging
from typing import Dict, Any, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load the model and return a predict function.
    Args:
        model_dir: Directory where model weights are stored
    Returns:
        Dictionary containing the model and predictor
    """
    try:
        from src.utils.inference import ISNetPredictor
        
        model_path = os.path.join(model_dir, 'model.pth')
        
        predictor = ISNetPredictor(
            model_path=model_path,
            input_size=(1024, 1024),
            device=None
        )
        
        return {
            'predictor': predictor
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def input_fn(request_body: bytes, request_content_type: str) -> Dict[str, Any]:
    """
    Deserialize and prepare the prediction input.
    Args:
        request_body: The request body
        request_content_type: The request content type
    Returns:
        Dictionary containing the input data and parameters
    """
    try:
        input_data = None
        input_size = (1024, 1024)
        output_format = 'png'
        
        if request_content_type == 'application/json':
            request = json.loads(request_body.decode('utf-8'))
            
            if 'parameters' in request:
                params = request.get('parameters', {})
                input_size = params.get('input_size', input_size)
                output_format = params.get('output_format', output_format)
            
            if 'image' in request:
                # Image is provided as base64
                image_bytes = base64.b64decode(request['image'])
                image = Image.open(io.BytesIO(image_bytes))
                input_data = np.array(image)
            elif 'image_url' in request:
                # Image is provided as URL
                import urllib.request
                from skimage import io as skio
                
                url = request['image_url']
                input_data = skio.imread(urllib.request.urlopen(url))
            else:
                raise ValueError("No 'image' or 'image_url' found in request")
            
        elif request_content_type.startswith('image/'):
            image = Image.open(io.BytesIO(request_body))
            input_data = np.array(image)
            
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
        
        return {
            'input_data': input_data,
            'input_size': input_size,
            'output_format': output_format
        }
        
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        raise


def predict_fn(input_data: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply model to the input data and return predictions.
    Args:
        input_data: Dictionary containing the input data and parameters
        model: Dictionary containing the model and predictor
    Returns:
        Dictionary containing the prediction results
    """
    try:
        predictor = model['predictor']
        
        if 'input_size' in input_data and input_data['input_size'] != predictor.input_size:
            predictor.input_size = input_data['input_size']
            
        prediction = predictor.predict(
            image=input_data['input_data']
        )
        
        return {
            'prediction': prediction,
            'output_format': input_data['output_format']
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def output_fn(prediction: Dict[str, Any], accept: str) -> Tuple[bytes, str]:
    """
    Serialize the prediction result.
    Args:
        prediction: Dictionary containing the prediction results
        accept: The accept content type
    Returns:
        Tuple of response body and content type
    """
    try:
        mask = prediction['prediction']
        output_format = prediction['output_format']
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(mask_uint8.squeeze())
            
        if accept.startswith('image/'):
            buffer = io.BytesIO()
            
            if output_format.lower() == 'png':
                pil_image.save(buffer, format='PNG')
                return buffer.getvalue(), 'image/png'
            elif output_format.lower() == 'jpeg' or output_format.lower() == 'jpg':
                pil_image.save(buffer, format='JPEG', quality=90)
                return buffer.getvalue(), 'image/jpeg'
            else:
                pil_image.save(buffer, format='PNG')
                return buffer.getvalue(), 'image/png'
        else:
            buffer = io.BytesIO()
            pil_image.save(buffer, format=output_format.upper())
            serialized_mask = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            response = {
                'mask': serialized_mask,
                'format': output_format
            }
            
            return json.dumps(response).encode('utf-8'), 'application/json'
            
    except Exception as e:
        logger.error(f"Error serializing output: {e}")
        raise 