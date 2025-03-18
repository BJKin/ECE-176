# Context Encoders: Feature Learning by Inpainting

This repository serves as a modern reimplementation the paper "[Context Encoders: Feature Learning by Inpainting](https://arxiv.org/abs/1604.07379)" by Deepak Pathak et al. in PyTorch.

## Project Proposal Summary

Image inpainting has long been a problem that researchers have tried to solve through a variety of methods over the years. It is the task of filling missing or corrupted pieces of an image with a computed version of that missing piece. In a paper entitled Context Encoders: Feature Learning by Inpainting, a new methodology for image inpainting was proposed through a modified convolutional neural network architecture the authors call Context Encoders. The main idea is to establish context around the missing section of an image by analyzing the surrounding components to generate the appropriate content for that missing region. In this project, we hope to reimplement the neural network architecture described in this paper using alternative training datasets while achieving similar results.

Now a staple in many modern smartphones, image inpainting is synonymous with editing and cropping out unsightly or undesirable aspects of an image. At a higher level, we are training a neural network to guess what should be filled in the missing piece using context, or surrounding pixels in this case. The solution mentioned in the paper describes a method that attempts to balance realism and pixel accuracy in the output.

The convolutional neural network described in the paper uses an encoder-decoder-style architecture with a channel-wise fully connected layer to connect the two and utilizes two separate loss functions in tandem to optimize the network. The first of the loss functions is a reconstruction loss that consists of a normalized masked L2 distance, while the other is an adversarial loss derived from a generative adversarial network. We have reimplemented the network using the modern PyTorch framework instead of the original Caffe and Torch implementation.

## Implementation Overview

This implementation provides two distinct models:

1. **Feature Learning Model** (`FeatureLearner.py`): A model trained using only reconstruction loss (L2), designed to learn general-purpose visual features that transfer well to other tasks like classification.

2. **Semantic Inpainting Model** (`SemanticInpainter.py`): A model trained with a combination of reconstruction and adversarial losses to produce high-quality, realistic image completions.

## Repository Structure

```
ECE-176/
├── Final-Project/
│   ├── data/               # Dataset storage
│   ├── losses/             # Loss function implementations
│   ├── models/             # Neural network architectures
│   │   ├── FeatureLearner.py     # Feature learning architecture with AlexNet classifier
│   │   └── SemanticInpainter.py  # Inpainting architecture with adversarial discriminator
│   ├── utilities/          # Helper functions and visualization utilities
│   └── contextEncoders.ipynb    # Jupyter notebook demonstrating the workflow
├── Images-and-Documents/
├── README.md           # This documentation file
└── requirements.txt    # Required packages
```

## Installation

### Prerequisites

- Python 3.10.16
- CUDA 12.6
- Anaconda/Miniconda

### Setting Up the Environment

1. **Clone this repository**:
   ```bash
   git clone https://github.com/BJKin/ECE-176.git
   cd ECE-176/Final-Project
   ```

2. **Create and activate a conda environment**:
   ```bash
   conda create -n context-encoders python=3.10.16
   conda activate context-encoders
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify CUDA compatibility**:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
   ```
## Running the Jupyter Notebook

The included Jupyter notebook (`contextEncoders.ipynb`) demonstrates the complete workflow:

```bash
jupyter notebook inpainting.ipynb
```

The notebook covers:
1. Data loading and preprocessing
2. Model initialization and configuration
3. Training procedures for both models
4. Visualization of inpainting results
5. Feature extraction examples

## Model Architecture Details

### Feature Learning Model (based on AlexNet)

1. **Encoder**:
   - Uses AlexNet architecture with 5 convolutional layers
   - Employs Leaky ReLU activations (negative slope = 0.2) instead of standard ReLU
   - Includes batch normalization after each layer
   - Final output size: 256 × 6 × 6

2. **Channel-wise Fully Connected Layer**:
   - Implemented as a depthwise convolution (groups=256) to connect feature maps
   - Followed by a 1×1 convolution to allow cross-channel information flow
   - Includes dropout with 0.5 rate to prevent overfitting

3. **Decoder**:
   - 5 transposed convolutional layers (upsampling)
   - Standard ReLU activations
   - Final bilinear upsampling to reach 227 × 227 output size

### Semantic Inpainting Model

1. **Encoder**:
   - 6 convolutional layers with increasing filter sizes
   - Leaky ReLU activations
   - Batch normalization
   - 4000-dimensional bottleneck

2. **Decoder**:
   - 5 transposed convolutional layers
   - ReLU activations (except final layer)
   - Tanh activation in the final layer to normalize output

3. **Discriminator**:
   - 5 convolutional layers
   - Leaky ReLU activations
   - Sigmoid output for binary classification (real/fake)

## Optimization Approaches

While the original paper used the stochastic gradient descent solver ADAM for optimization purposes, we used a more modern approach by combining ADAMW with a 1cycle learning rate scheduler to improve performance. In brief, we reimplement the exact neural network architecture described in Pathak et al., but utilize modern tools for ease of implementation and to enhance performance where possible.

## Dataset and Experiments

For this implementation, we use a dataset composed of 12,000 images from the Kaggle landscape recognition dataset. The data set is composed of images that are divided into five distinct categories of landscapes: coast, desert, forest, glaciers, and mountains. For each image, a duplicate is created for which a certain amount of pixels are blocked out (image masking), and each pair is fed as training data to the network.

The dataset is already split into training, validation, and testing groups. We also experiment with various sizes and locations of image masking, to see what effect, if any, it has on the final weights and biases. Not all of the images are the same size, so we manually resize the images to fixed dimensions before training and testing.

## Citation

If you use this implementation, please cite the original paper:

```
@misc{pathak2016contextencodersfeaturelearning,
      title={Context Encoders: Feature Learning by Inpainting}, 
      author={Deepak Pathak and Philipp Krahenbuhl and Jeff Donahue and Trevor Darrell and Alexei A. Efros},
      year={2016},
      eprint={1604.07379},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1604.07379}, 
}
```

## License

This project is released under the MIT License.
