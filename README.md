# Neural Architecture Search (NAS) Project

## Description
This project aims to automate the process of neural architecture search (NAS) for optimizing convolutional neural networks (CNNs) for image recognition tasks. The Tree-structured Parzen Estimator (TPE) strategy is utilized to optimize the network architectures. Microsoft’s NNI framework is employed for streamlining hyperparameter tuning and architecture search, thus enhancing productivity and performance.

The goal is to automatically discover and fine-tune architectures that outperform manually designed networks. The project demonstrates the AutoML capability to automatically select optimal parameters, reducing manual intervention and accelerating model development. It is tested and validated on the MNIST dataset, showcasing its applicability.

## Code
The main code file `nas_project.py` includes the implementation of the NAS algorithm, utilizing TensorFlow and Keras for building and training CNNs. Below are the key components of the code:

- **Data Loading:** MNIST dataset is loaded using TensorFlow Datasets (TFDS).
- **Model Creation:** A CNN model is created with configurable hyperparameters.
- **Training Loop:** The model is trained using the provided training dataset with configurable parameters such as batch size, learning rate, etc.
- **Hyperparameter Tuning:** Microsoft's NNI framework is used to perform hyperparameter tuning and architecture search automatically.

## Usage
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python nas_project.py
   ```
4. The script will execute the neural architecture search process and report the final test accuracy.

## Requirements
- Python 3
- TensorFlow
- Keras
- TensorFlow Datasets (TFDS)
- NNI framework


