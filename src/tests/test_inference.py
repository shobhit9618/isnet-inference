"""
Tests for the inference functionality
"""
import os
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from src.utils.inference import ISNetPredictor


# Create a simple test model class
class MockISNetModel(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(MockISNetModel, self).__init__()
        self.conv = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)
        self.side1 = nn.Conv2d(64, out_ch, kernel_size=1)
        
    def forward(self, x):
        feat = self.conv(x)
        
        h, w = x.shape[2], x.shape[3]
        outputs = [torch.ones(x.shape[0], 1, h, w) * 0.5 for _ in range(6)]
        
        features = [feat for _ in range(6)]
        
        return outputs, features
    
    def get_final_output(self, x):
        h, w = x.shape[2], x.shape[3]
        return torch.ones(x.shape[0], 1, h, w) * 0.5


# Create a pytest fixture for a temporary model file
@pytest.fixture
def temp_model_path():
    """Create a temporary model file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        model = MockISNetModel()
        torch.save(model.state_dict(), tmp.name)
        model_path = tmp.name
    
    yield model_path
    
    if os.path.exists(model_path):
        os.remove(model_path)


# Monkeypatch ISNetDIS
@pytest.fixture
def monkeypatch_isnetdis(monkeypatch):
    import src.utils.inference
    
    monkeypatch.setattr('src.models.ISNetDIS', MockISNetModel)
    
    yield


@pytest.fixture
def test_images():
    return {
        "rgb": np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
        "grayscale": np.random.randint(0, 256, (100, 100), dtype=np.uint8),
        "single_channel": np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8),
        "rgba": np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
    }


@pytest.fixture
def temp_dirs():
    temp_input = tempfile.mkdtemp()
    temp_output = tempfile.mkdtemp()
    
    yield temp_input, temp_output
    
    shutil.rmtree(temp_input, ignore_errors=True)
    shutil.rmtree(temp_output, ignore_errors=True)


# Monkeypatch skimage.io
@pytest.fixture
def monkeypatch_io(monkeypatch, test_images):
    class MockIO:
        @staticmethod
        def imread(path):
            return test_images["rgb"]
        
        @staticmethod
        def imsave(path, arr):
            pass
    
    monkeypatch.setattr('skimage.io', MockIO)
    
    yield


class TestISNetPredictor:
    
    def test_init(self, temp_model_path, monkeypatch_isnetdis):
        predictor = ISNetPredictor(temp_model_path)
        
        assert predictor.model is not None
        assert predictor.input_size == (1024, 1024)
        assert predictor.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_preprocess_image(self, temp_model_path, monkeypatch_isnetdis, test_images):
        predictor = ISNetPredictor(temp_model_path)
        
        # Test RGB image
        preprocessed = predictor.preprocess_image(test_images["rgb"])
        assert isinstance(preprocessed, torch.Tensor)
        assert preprocessed.shape[0] == 1  # batch size
        assert preprocessed.shape[1] == 3  # channels
        assert preprocessed.shape[2:] == predictor.input_size  # height, width
        
        # Test grayscale image
        preprocessed = predictor.preprocess_image(test_images["grayscale"])
        assert preprocessed.shape[1] == 3
        
        # Test single-channel image
        preprocessed = predictor.preprocess_image(test_images["single_channel"])
        assert preprocessed.shape[1] == 3
        
        # Test RGBA image
        preprocessed = predictor.preprocess_image(test_images["rgba"])
        assert preprocessed.shape[1] == 3
    
    def test_predict(self, temp_model_path, monkeypatch_isnetdis, test_images):
        predictor = ISNetPredictor(temp_model_path)
        
        prediction = predictor.predict(test_images["rgb"])
        
        assert isinstance(prediction, np.ndarray)
        assert prediction.shape[:2] == test_images["rgb"].shape[:2]
        assert prediction.shape[2] == 1