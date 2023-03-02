# athena
This repository focuses  on the project structure rather than the accuracy of code implementation. It contains a PyTorch implementation of a deep learning model for image classification. In this README, we provide instructions for how to use the model, and any limitations to its performance.

# Model
This Python script models.py implements a factory design pattern for creating PyTorch classification models of different sizes.
The script defines four different models:

- `TinyModel`: Input size of 32x32
- `SmallModel`: Input size of 64x64
- `LargeModel`: Input size of 128x128
- `BaseModel`: Input size of 256x256.

Each model is a PyTorch `nn.Module` and has a `predict()` method for making predictions on input data.


# How to Use
To use this script, simply import the `ModelFactory` class and use it to create a model:

```python
from models import ModelFactory

model_factory = ModelFactory()

# create a TinyModel with 5 output classes
model = model_factory.create('tiny', 5)

# create a LargeModel with 10 output classes
model = model_factory.create('large', 10)
```
# Installation
1. Clone the repository: `git clone https://github.com/salahali2019/athena_ai.git`
2. Navigate to the repository directory: `cd athena_ai`
3. Install the package: `pip install .`

# Limitations

* The images should all have the same size.
   This can be improved in the next release by change the way the images are saved. Arrays could be stored in h5 arrays where it allows for different sizes
* the models handles fixed input size, TinyModel : 32x32, SmallModel : 64x64, and LargeModel :256x256
   This can be improved in the next release by adding a transform function that change the size of any input to this fixed size
