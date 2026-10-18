import math
import random

inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
targets = (0, 1, 1, 0)

weights_hidden = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
weights_output = [random.uniform(-1, 1) for _ in range(2)]

biases_hidden = [0, 0]
bias_output = 0

def train(epochs: int = 1000, learning_rate: int = 0.5) -> None:
    for epoch in range(1, epochs + 1):
        z_hidden = []
        for input in inputs:
            z = []
            z.append(weights_hidden[0][0] * input[0] +
                     weights_hidden[0][1] * input[1] +
                     biases_hidden[0])
            z.append(weights_hidden[1][0] * input[0] +
                     weights_hidden[1][1] * input[1] +
                     biases_hidden[1])
            z_hidden.append(z)
    print(f"Z hidden: {z_hidden}")
    print("--------------------------------------------------------")
                

def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))

def relu(z: float) -> float:
    return 1 if z > 0 else 0

def predict() -> list[int]:
    pass

def main() -> None:
    print("Initial Weights: \n" \
         f"Hidden Weights: {weights_hidden} \n" \
         f"Output Weights: {weights_output} \n" \
          "--------------------------------------------------------")
    train(epochs=1)

if __name__ == "__main__":
    main()