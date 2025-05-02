import numpy as np
import matplotlib.pyplot as plt

class HebbianNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        """
        Initialize the Hebbian network
        :param input_size: Size of the input layer
        :param output_size: Size of the output layer
        :param learning_rate: Learning rate
        """
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.learning_rate = learning_rate
        
    def train(self, input_pattern, output_pattern, epochs=1):
        """
        Train the network
        :param input_pattern: Input pattern
        :param output_pattern: Output pattern
        :param epochs: Number of training epochs
        """
        input_pattern = np.array(input_pattern)
        output_pattern = np.array(output_pattern)
        
        for _ in range(epochs):
            # Hebbian learning rule: Δw = η * x * y
            delta_w = self.learning_rate * np.outer(input_pattern, output_pattern)
            self.weights += delta_w
            
    def predict(self, input_pattern):
        """
        Predict the output
        :param input_pattern: Input pattern
        :return: Output activation values
        """
        input_pattern = np.array(input_pattern)
        return np.dot(input_pattern, self.weights)
    
    def visualize_weights(self):
        """Visualize the weight matrix"""
        plt.figure(figsize=(8, 6))
        plt.imshow(self.weights, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Hebbian Learning Weight Matrix')
        plt.xlabel('Output Neurons')
        plt.ylabel('Input Neurons')
        plt.show()

# Example: Learning simple letter associations
def example_letter_association():
    # Define input patterns (simple letter representations)
    # Each letter is represented by a 5x5 binary matrix
    A = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1]
    ]).flatten()
    
    B = np.array([
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0]
    ]).flatten()
    
    C = np.array([
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1]
    ]).flatten()
    
    # Create the network (25 input neurons, 3 output neurons for A, B, C)
    network = HebbianNetwork(input_size=25, output_size=3, learning_rate=0.1)
    
    # Train the network to associate letters with categories
    # Output patterns: A=[1,0,0], B=[0,1,0], C=[0,0,1]
    network.train(A, [1, 0, 0], epochs=5)
    network.train(B, [0, 1, 0], epochs=5)
    network.train(C, [0, 0, 1], epochs=5)
    
    # Test the network
    test_pattern = np.array([
        [0, 1, 1, 1, 0],  # Similar to A but with noise
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 1]  # Slightly different at the end
    ]).flatten()
    
    print("Original A pattern output:", network.predict(A))
    print("Original B pattern output:", network.predict(B))
    print("Original C pattern output:", network.predict(C))
    print("Test pattern output:", network.predict(test_pattern))
    
    # Visualize the weights
    network.visualize_weights()

# Example: Learning color and text associations
def example_color_association():
    # Define input patterns (RGB colors + text descriptions)
    # Each color is represented by 3 values (RGB) + 5 text features
    red = np.array([1, 0, 0, 1, 0, 0, 0, 0])  # RGB=1,0,0 + features for the word "red"
    green = np.array([0, 1, 0, 0, 1, 0, 0, 0])
    blue = np.array([0, 0, 1, 0, 0, 1, 0, 0])
    color_names = np.array([0, 0, 0, 1, 1, 1, 0, 0])  # Text features
    
    # Create the network (8 input neurons, 3 output neurons for red, green, blue)
    network = HebbianNetwork(input_size=8, output_size=3, learning_rate=0.1)
    
    # Train the network
    network.train(red, [1, 0, 0], epochs=10)
    network.train(green, [0, 1, 0], epochs=10)
    network.train(blue, [0, 0, 1], epochs=10)
    network.train(color_names, [0.5, 0.5, 0.5], epochs=5)  # Associate text with all colors
    
    # Test the network
    print("\nColor association test:")
    print("Red input output:", network.predict(red))
    print("Green input output:", network.predict(green))
    print("Blue input output:", network.predict(blue))
    print("Color names output:", network.predict(color_names))
    
    # Test partial information
    test_color = np.array([0.8, 0.2, 0.1, 0, 0, 0, 0, 0])  # Mostly red
    print("Test color output:", network.predict(test_color))
    
    # Visualize the weights
    network.visualize_weights()

if __name__ == "__main__":
    print("Letter association example:")
    example_letter_association()
    
    print("\nColor association example:")
    example_color_association()