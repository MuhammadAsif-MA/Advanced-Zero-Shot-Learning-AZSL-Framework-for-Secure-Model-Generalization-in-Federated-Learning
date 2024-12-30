# Advanced Zero-Shot Learning (AZSL) Framework

This repository contains the implementation of the AZSL framework for federated learning with Zero-Shot Learning (ZSL). The project includes synthetic data generation using WGAN-GP, global model training with EfficientNet-B7, and integration with TensorFlow Federated.

## Features
- Federated learning with real-world simulation.
- Zero-Shot Learning for unseen class classification.
- Synthetic data generation with WGAN-GP.
- Visualization of learning curves and generalization gap.

## Requirements
- Python 3.8+
- TensorFlow, TensorFlow Federated
- EfficientNet-PyTorch, Matplotlib, NumPy, etc.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/azsl-framework.git
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Navigate to the `scripts` directory:
   ```
   cd scripts
   ```
2. Run the main script:
   ```
   python azsl_framework.py
   ```

## License
This project is licensed under the MIT License.
