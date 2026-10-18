import math
import random

inputs = ((0,0),(0,1),(1,0),(1,1))
targets = (0, 1, 1, 1)
weights = ()
bias = 0

def init() -> None:
    global weights
    weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
    print(f"Initial weights: {weights}")

def sigmoid(z: list[float]) -> float:
    return [1 / (1 + math.exp(-x)) for x in z]

def train() -> None:
    for epoch in range(1):
        z = [x[0] * weights[0] + x[1] * weights[1] + bias for x in inputs]
        a = sigmoid(z)
        print(f"z: {z}")
        print(f"a: {a}")
        cost = sum((prediction - target) ** 2
               for prediction, target, in zip(a, targets)) / 4
        print(f"Cost: {cost}")
        

def main() -> None:
    init()
    train()

if __name__ == "__main__":
    main()