import os
import urllib.request
import numpy as np
from PIL import Image
import scipy.special
import matplotlib.pyplot as plt
import scipy.ndimage

class NeuralNetwork:
    """
    A 3-layer neural network (Input, Hidden, Output)
    """
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Link weight matrices, wih (weight input->hidden) and who (weight hidden->output).
        # Weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        # Learning rate
        self.lr = learningrate
        
        # Activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        """Train the neural network"""
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Output layer error is the (target - actual)
        output_errors = targets - final_outputs

        # Hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
    def query(self, inputs_list):
        """Query the neural network"""
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    def backquery(self, targets_list):
        """
        Backquery the neural network to visualize what it thinks a digit looks like
        """
        final_outputs = np.array(targets_list, ndmin=2).T
        
        # Restrict values to avoid errors in logit
        final_outputs = np.clip(final_outputs, 0.01, 0.99)
        # Inverse of sigmoid function (logit)
        final_inputs = scipy.special.logit(final_outputs)

        # Calculate signals emerging from hidden layer
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # Scale back to 0.01 - 0.99
        hidden_outputs = (hidden_outputs - hidden_outputs.min()) / (hidden_outputs.max() - hidden_outputs.min())
        hidden_outputs = hidden_outputs * 0.98 + 0.01
        
        # Inverse of sigmoid function (logit)
        hidden_inputs = scipy.special.logit(hidden_outputs)

        # Calculate signals emerging from input layer
        inputs = np.dot(self.wih.T, hidden_inputs)
        # Scale back to 0.01 - 0.99
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
        inputs = inputs * 0.98 + 0.01
        
        return inputs
    
def train_with_augmentation(network, image_array, label):
    """Train the network with original and rotated images (±10 degrees)"""
    inputs = (image_array / 255.0 * 0.99) + 0.01
    inputs = inputs.reshape(784)
    targets = np.zeros(10) + 0.01
    targets[label] = 0.99
    
    # Train on original image
    network.train(inputs, targets)
    
    # Train on rotated images
    for angle in [-10, 10]:
        rotated = scipy.ndimage.rotate(image_array, angle, reshape=False, order=1, mode='nearest')
        inputs_rot = (rotated / 255.0 * 0.99) + 0.01
        inputs_rot = inputs_rot.reshape(784)
        network.train(inputs_rot, targets)

def safe_recognize_digit(network, filename):
    """Safely recognize a single PNG image"""
    try:
        img = Image.open(filename)
    except FileNotFoundError:
        print(f"File {filename} not found!")
        return None
    except Exception as e:
        print(f"Error opening {filename}: {e}")
        return None

    # Optional: Plot the original image
    # plt.imshow(img, cmap='Greys'); plt.title(f"Original {filename}"); plt.show()
    
    # Convert to grayscale and resize to 28x28
    img = img.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
    # Invert colors if necessary (assuming dark digit on light background)
    img = Image.eval(img, lambda x: 255 - x)
    img_array = np.array(img)

    img_norm = (img_array / 255.0 * 0.99) + 0.01
    result = network.query(img_norm.reshape(784))
    
    plt.imshow(img_array, cmap='Greys')
    plt.title(f"Preprocessed {filename}\\nPredicted: {np.argmax(result)}")
    plt.show()
    
    print(f"[{filename}] Network outputs:", result.ravel())
    print(f">>> Predicted: {np.argmax(result)}")
    return np.argmax(result)

def ensure_dataset(filename, url):
    """Check if the dataset exists, otherwise download it."""
    if not os.path.exists(filename):
        print(f"Dataset {filename} not found. Downloading from {url}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            print(f"Please manually download the file and save it as {filename}")
            raise SystemExit

def main():
    # ======= Network parameters and initialization =======
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.1
    epochs = 5 # Set epochs for training

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # ======= Ensure datasets exist =======
    train_file = "mnist_train_60K.csv"
    test_file = "mnist_test10K.csv"
    
    # URLs for the MNIST CSV dataset
    train_url = "https://pjreddie.com/media/files/mnist_train.csv"
    test_url = "https://pjreddie.com/media/files/mnist_test.csv"

    ensure_dataset(train_file, train_url)
    
    # ======= Load training dataset =======
    try:
        with open(train_file, 'r') as f:
            data_list = f.readlines()
        print(f"Loaded {len(data_list)} training samples")
    except FileNotFoundError:
        raise SystemExit(f"File {train_file} not found!")

    # ======= Training process =======
    print("Starting training...")
    for e in range(epochs):
        total_samples = len(data_list)
        for i, record in enumerate(data_list):
            all_values = record.split(',')
            label = int(all_values[0])  # Correct answer (digit label)
            image_array = np.asarray(all_values[1:], dtype=float).reshape((28, 28))
            
            # Train with data augmentation
            train_with_augmentation(nn, image_array, label)
            
            if i > 0 and i % 5000 == 0:
                progress = (i / total_samples) * 100
                print(f"Epoch {e+1}, Progress: {progress:.1f}% ({i}/{total_samples})")
        print(f"Epoch {e+1} complete")

    # ======= Testing process =======
    ensure_dataset(test_file, test_url)
    try:
        with open(test_file, 'r') as f:
            test_list = f.readlines()
    except FileNotFoundError:
        print(f"Test file {test_file} is missing. Skipping accuracy calculation.")
        test_list = []

    if test_list:
        scorecard = []
        for record in test_list:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
            outputs = nn.query(inputs)
            label = np.argmax(outputs)
            scorecard.append(1 if label == correct_label else 0)
            
        scorecard_array = np.asarray(scorecard)
        print(f"Accuracy on test set: {scorecard_array.mean() * 100:.2f}%")

    # ======= Fine-tuning on custom png images =======
    print("Fine-tuning on custom images (1.png to 4.png)...")
    for i in range(1, 5):
        try:
            if os.path.exists(f"{i}.png"):
                img = Image.open(f"{i}.png").convert('L').resize((28, 28))
                train_with_augmentation(nn, np.array(img), i if i < 10 else 0)
        except Exception as ex:
            print(f"Error with file {i}.png: {ex}")

    # ======= Test custom images =======
    print("Recognizing custom images...")
    for i in range(1, 5):
        if os.path.exists(f"{i}.png"):
            safe_recognize_digit(nn, f"{i}.png")

    # ======= Visualize \"backquery\" (Neural network's perception of numbers) =======
    print("Generating backquery visualizations...")
    for i in range(10):
        target_outputs = np.zeros(10) + 0.01
        target_outputs[i] = 0.99
        img = nn.backquery(target_outputs)
        plt.imshow(img.reshape(28, 28), cmap='Greys')
        plt.title(f'How the network sees the digit {i}')
        plt.show()

if __name__ == "__main__":
    main()