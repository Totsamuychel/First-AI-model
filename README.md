# 🧠 First AI Model (MNIST Neural Network from Scratch)

A lightweight, purely mathematical implementation of a 3-layer Artificial Neural Network (ANN) from scratch using Python. This project was created to understand the fundamental math behind neural networks without relying on heavy machine learning frameworks like TensorFlow or PyTorch.

The model is trained on the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/) to classify handwritten digits (0-9).

## 🌟 Features
- **Built from Scratch:** Core math logic written entirely using `numpy` (forward and backward propagation).
- **Data Augmentation:** Automatically augments training data by rotating images ±10 degrees to improve robustness and prevent overfitting.
- **Backquerying:** Reverses the neural network to visually see what the network "thinks" each digit looks like.
- **Custom Image Testing:** Can safely load, preprocess, and predict your own hand-drawn digits (e.g., `1.png`, `2.png`).
- **Auto-Dataset Download:** Automatically downloads the required MNIST CSV datasets if they are not present locally.

## ⚙️ How It Works
The architecture of this neural network is simple yet effective:
- **Input Layer:** 784 nodes (corresponding to the 28x28 pixels of an MNIST image).
- **Hidden Layer:** 100 nodes (adjustable parameter).
- **Output Layer:** 10 nodes (representing digits 0 through 9).
- **Activation Function:** Sigmoid function.
- **Loss Optimization:** Gradient Descent (via Backpropagation).

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Totsamuychel/First-AI-model.git
cd First-AI-model
```

### 2. Install Dependencies
It is highly recommended to use a virtual environment. Install the dependencies via `pip`:
```bash
pip install -r requirements.txt
```

### 3. Run the Model
Simply run the main python script:
```bash
python AI_model.py
```
*Note: On its first run, the script will automatically attempt to download the `mnist_train_60K.csv` and `mnist_test10K.csv` datasets (approx. 120MB total). Please be patient while it downloads.*

### 4. Provide Custom Images
To test the network on your own handwriting:
1. Create a 28x28 pixel square image.
2. Draw a digit in the center (black digit on a white background).
3. Save it as `1.png`, `2.png`, `3.png`, or `4.png` in the root folder.
4. Run the script! The network will fine-tune itself on your custom images and then predict the digits, popping up a `matplotlib` window showing its processing steps.

## 📦 Requirements
- `numpy` - For matrix multiplications and vector math.
- `Pillow` (PIL) - For reading and preprocessing custom `.png` images.
- `scipy` - For the sigmoid activation function and image rotation.
- `matplotlib` - For plotting the images and backquery results.

## 🧠 What is "Backquerying"?
Once the network is trained, we can pass an ideal output (e.g., `[0.01, 0.99, 0.01, ...]` which represents a perfect `1`) backwards through the network to generate an image. This reveals the "mental model" or abstract representation the neural network has developed for each digit during training!
