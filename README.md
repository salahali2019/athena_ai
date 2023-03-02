# athena_ai
This repository contains a PyTorch implementation of a deep learning model for image classification. In this README, we provide instructions for models, how to use the model, ,and any limitations to its performance.

# Model
The model is a convolutional neural network that has been trained on the ImageNet dataset for image classification. The model takes a single input image and produces a vector of probabilities indicating the likelihood that the image belongs to each of the 1000 ImageNet classes.

# How to Use
To use the model for inference on your own images, follow these steps:

Clone the repository: git clone https://github.com/your-username/your-repo.git
Download the pre-trained weights for the model from this link: [insert link here]
Place the weights file in the models directory of the repository
Install the required Python packages by running: pip install -r requirements.txt
Run the inference.py script with the path to your input image as an argument, like this: `python inference.py path

# Limitations

* The images should all have the same size.
   This can be improved in the next release by change the way the images are saved. Arrays could be stored in h5 arrays where it allows for different sizes
* the models handles fixed input size, TinyModel : 32x32, SmallModel : 64x64, and LargeModel :256x256
   This can be improved in the next release by adding a transform function that change the size of any input to this fixed size
