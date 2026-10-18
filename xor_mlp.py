import math
import random

inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
targets = (0, 1, 1, 0)

weights_hidden = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
weights_output = [random.uniform(-1, 1) for _ in range(2)]

biases_hidden = [0, 0]
bias_output = 0

def train(epochs: int = 1000, learning_rate: int = 0.5) -> None:
    global weights_hidden, weights_output, biases_hidden, bias_output
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
        a_hidden = [[relu(x) for x in z] for z in z_hidden]
        print(f"A hidden: {a_hidden}")
        print("--------------------------------------------------------")
        z_output = [weights_output[0] * input[0] + 
                    weights_output[1] * input[1] +
                    bias_output
                    for input in inputs]
        print(f"Z output: {z_output}")
        print("--------------------------------------------------------")
        a_output = [sigmoid(z) for z in z_output]
        print(f"A output: {a_output}")
        print("--------------------------------------------------------")

        dw_hidden = [[0, 0], [0, 0]] 
        db_hidden = [0, 0]

        dw_output = [0, 0]
        db_output = 0

        for i, (input, target) in enumerate(zip(inputs, targets)):
            prediction = a_output[i]
            d_error = (prediction - target) * prediction * (1 - prediction)
            dw_output = [dw + (d_error * x) for dw, x in zip(dw_output, input)]

def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))

def relu(z: float) -> float:
    return z if z > 0 else 0

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