import numpy as np
import math

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

weights_hidden = np.random.uniform(-math.sqrt(1/2), math.sqrt(1/2), (2, 2))
weights_output = np.random.uniform(-math.sqrt(1/2), math.sqrt(1/2), (1, 2))

def train(epochs: int = 1000, learning_rate: float = 0.5) -> None:
    pass

def predict() -> list[int]:
    pass

def main() -> None:
    pass

if __name__ == "__main__":
    main()