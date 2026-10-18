import math
import random

inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
targets = (0, 1, 1, 0)

weights_hidden = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
weights_output = [random.uniform(-1, 1) for _ in range(2)]

bias_hidden = [0, 0]
bias_output = 0

def train(epochs: int = 1000, learning_rate: int = 0.5) -> None:
    pass

def predict() -> list[int]:
    pass

def main() -> None:
    pass

if __name__ == "__main__":
    main()