import os
from pathlib import Path

from src.utils.inference import ISNetPredictor

def main():
    model_path = "/Users/shobhitgupta/Documents/codes/kittl_assignment/src/models/isnet-general-use.pth"
    image_path = "/Users/shobhitgupta/Pictures/my_photo.jpg"
    output_dir = "outputs"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing predictor with model: {model_path}")
    predictor = ISNetPredictor(model_path)
    
    print(f"Running inference on image: {image_path}")
    mask = predictor.predict_from_file(image_path)
    predictor.save_prediction(mask, output_dir)
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main() 