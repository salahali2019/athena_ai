import torch
from models import ModelFactory
from utils import inference, plot_confusion_matrix, plot_calibration_curve, save_false_positives, find_false_positive_patterns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torchvision.datasets as datasets
import argparse

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Testing the classification models')    
    parser.add_argument('data', default='CIFAR10', help='path to input data file')
    parser.add_argument('--model', default='tiny', choices=['tiny', 'small', 'large', 'base'], help='name of model to use')
    args = parser.parse_args()
    
    # Create a model using the factory pattern
    factory = ModelFactory()
    model = factory.create_model(args.model)
    
    # Load dataset             
    if args.data=='CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        image_data=dataset.data
        image_target=dataset.targets
        classes=data.classes
        preds= inference(model, image_data)
        plot_confusion_matrix(image_target, preds, classes)
    else:    
        folder_name=args.data
        images = []
        for filename in os.listdir(folder_name):
            img = Image.open(os.path.join(folder_name, filename))
            images.append(np.array(img)
         image_data = np.vstack(images)
         preds= inference(model, image_data)
                                  
    
