import math
import random

inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
targets = (0, 1, 1, 1)
weights = []
bias = 0

def init() -> None:
    global weights
    weights = [random.uniform(-1, 1), random.uniform(-1, 1)]

def sigmoid(z: list[float]) -> float:
    return [1 / (1 + math.exp(-x)) for x in z]

def train(epochs=1000) -> None:
    global weights, bias
    for epoch in range(1, epochs + 1):
        z = [x[0] * weights[0] + x[1] * weights[1] + bias for x in inputs]
        a = sigmoid(z)
        cost = sum((prediction - target) ** 2
               for prediction, target, in zip(a, targets)) / 8

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Cost: {cost}")
            print(f"Weights: {weights} | Bias: {bias}")
            print("----------------------------------"
             "---------------------------------------")

        dw1 = 0.0
        dw2 = 0.0
        db = 0.0

        for i, (input, target) in enumerate(zip(inputs, targets)):
            prediction = a[i]
            d_cost = prediction - target
            da = prediction * (1 - prediction)
            dw1 += d_cost * da * input[0]
            dw2 += d_cost * da * input[1]
            db += d_cost * da

        weights[0] -= 0.1 * (dw1 / 4)
        weights[1] -= 0.1 * (dw2 / 4)
        bias -= 0.1 * (db / 4)

def predict() -> list[int]:
    predictions = []
    for input in inputs:
        z = weights[0] * input[0] + weights[1] * input[1] + bias
        prediction = 1 if z >= 0.5 else 0
        predictions.append(prediction)
        print(f"Input: {input}")
        print(f"Prediction: {prediction}")
        print("----------------------------")

    if predictions == [0, 1, 1, 1]:
        print("IT WORKS!!!!!!!!!!!")
    return predictions

def main() -> None:
    print("-----------------------------------------------------")
    init()
    print("Initial Predictions:")
    predict()
    print("\n")
    train(2000)
    print("\n")
    print("Final Predictions")
    predict()

if __name__ == "__main__":
    main()