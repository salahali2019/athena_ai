import unittest
import torch

from my_module import ModelFactory

class TestModelFactory(unittest.TestCase):
    
    def test_create_tiny_model(self):
        # Test creating an invalid model type
        model = ModelFactory.create_model('tiny')

    
    def test_create_small_model(self):
        # Test creating an small model type
        model = ModelFactory.create_model('small')

    def test_create_large_model(self):
      # Test creating an large model type
      model = ModelFactory.create_model('large')

    def test_invalid_model_type(self):
        # Test creating an invalid model type      
        with self.assertRaises(ValueError):
            model = ModelFactory.create_model('invalid')
