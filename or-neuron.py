import math
import random

weights = ()
bias = 0

def init() -> None:
    global weights
    weights = [random.uniform(-1, 1), random.uniform(-1, 1)]

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(x))

def main():
    pass

if __name__ == "__main__":
    main()