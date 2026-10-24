import numpy as np
import numpy.typing as npt
import math

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

weights_hidden = np.random.uniform(-math.sqrt(1/2), math.sqrt(1/2), (2, 2))
weights_output = np.random.uniform(-math.sqrt(1/2), math.sqrt(1/2), (1, 2))

biases_hidden = np.array([[0.1], [0.1]])
bias_output = np.array([[0.1]])

def train(epochs: int = 1000, learning_rate: float = 0.5) -> None:
    for epoch in range(1):
        dw_hidden = np.zeros((2, 2))
        db_hidden = np.zeros((1, 2))

        dw_output = np.zeros((1, 2))
        db_output = 0

        for i, (x, target) in enumerate(zip(inputs, targets)):
            z_hidden = inputs @ weights_hidden.T + biases_hidden.T
            a_hidden = sigmoid(z_hidden)

            z_output = a_hidden @ weights_output.T + bias_output
            a_output = sigmoid(z_output)

            prediction = a_output[i]
            d_error = (prediction - target) * (prediction * (1 - prediction))

            dw_hidden += ((d_error * weights_output * a_hidden[i] * (1 - a_hidden[i])) @ x).T

            

def sigmoid(x: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-x))

def predict() -> list[int]:
    return []

def main() -> None:
    train()

if __name__ == "__main__":
    main()