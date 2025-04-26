import os
import time
import numpy as np
import logging
from typing import Union, Tuple, List, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ISNetPredictor:
    """
    Predictor class for ISNet segmentation model.
    
    Attributes:
        model: The ISNet model instance
        device: The device to use for inference (CPU or GPU)
        input_size: The input size for the model
    """
    
    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (1024, 1024),
        device: Optional[str] = None
    ):
        self.input_size = input_size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        from src.models import ISNetDIS
        self.model = ISNetDIS()
        
        try:
            state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess an image for model inference.
        Args:
            image: Input image as numpy array (H, W, C)
        Returns:
            Preprocessed image tensor (1, C, H, W)
        """
        # Handle grayscale images
        # Repeat the channel to make it 3-channel
        if len(image.shape) < 3:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
            
        # Convert to tensor
        im_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(
            torch.unsqueeze(im_tensor, 0), 
            self.input_size, 
            mode="bilinear"
        ).type(torch.float32)
        
        # Normalize
        image_normalized = torch.divide(im_tensor, 255.0)

        # Apply normalization with mean and std
        image_normalized = normalize(
            image_normalized, 
            [0.5, 0.5, 0.5], 
            [1.0, 1.0, 1.0]
        )
        
        return image_normalized
    
    @torch.no_grad()
    def predict(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Run inference on an image.
        Args:
            image: Input image as numpy array (H, W, C)
        Returns:
            Segmentation mask as numpy array (H, W, 1)
        """
        try:
            original_shape = image.shape[:2]
            
            preprocessed = self.preprocess_image(image).to(self.device)
            
            start_time = time.time()
            output = self.model.get_final_output(preprocessed)
            inference_time = time.time() - start_time
            
            # Resize to original size
            output_resized = F.interpolate(
                output, 
                size=original_shape, 
                mode='bilinear'
            )
            
            output_np = output_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            logger.info(f"Inference completed in {inference_time:.3f}s")
            return output_np
                
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
    
    def predict_from_file(
        self,
        image_path: Union[str, Path]
    ) -> np.ndarray:
        """
        Load an image from file and run inference.
        Args:
            image_path: Path to the input image
        Returns:
            Segmentation mask as numpy array
        """
        try:
            from skimage import io
            
            image = io.imread(image_path)
            
            return self.predict(image)
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def save_prediction(
        self,
        mask: np.ndarray,
        output_dir: str
    ) -> None:
        try:
            import io
            from PIL import Image
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Scale to [0, 255]
            mask = (mask * 255).astype(np.uint8)

            pil_image = Image.fromarray(mask.squeeze())

            buffer = io.BytesIO()
        
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            mask_data = buffer.getvalue()
            
            output_path = os.path.join(output_dir, "mask.png")
            with open(output_path, 'wb') as f:
                f.write(mask_data)
            
            logger.info(f"Saved prediction to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving prediction to {output_dir}: {e}")
            raise