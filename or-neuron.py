import math
import random

inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
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
    for epoch in range(1000):
        z = [x[0] * weights[0] + x[1] * weights[1] + bias for x in inputs]
        a = sigmoid(z)
        print(f"z: {z}")
        print(f"a: {a}")
        cost = sum((prediction - target) ** 2
               for prediction, target, in zip(a, targets)) / 8
        print(f"Cost: {cost}")

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Cost: {cost}")
            print(f"Weights: {weights}")

        dw1 = 0.0
        dw2 = 0.0
        db = 0.0

        for i, input, target in enumerate(zip(inputs, targets)):
            prediction = a[i]
            d_cost = (prediction - target)
            da = [(x)(1-x) for x in a]
            dw1 += d_cost * da * input[0]
            dw2 += d_cost * da * input[1]
            db += d_cost * da

        weights[0] -= 0.1 * (dw1 / 4)
        weights[1] -= 0.1 * (dw2 / 4)
        bias -= 0.1 * (db / 4)

        

def main() -> None:
    init()
    train()

if __name__ == "__main__":
    main()