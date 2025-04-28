import numpy as np
import sys

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward_propagate(x, Theta1, Theta2):
    a1 = np.insert(x, 0, 1)
    z2 = Theta1 @ a1
    a2 = np.insert(sigmoid(z2), 0, 1)
    z3 = Theta2 @ a2
    a3 = sigmoid(z3)
    return a1, z2, a2, z3, a3

def main(Theta1, Theta2, X, y, lambda_reg=0.0):
    X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    print(f"Regularization parameter lambda={lambda_reg:.3f}\n")

    layers = [X.shape[1], Theta1.shape[0], Theta2.shape[0]]
    layers_str = "[" + " ".join(str(layer) for layer in layers) + "]"
    print(f"Initializing the network with the following structure (number of neurons per layer): {layers_str}\n")

    print("Initial Theta1 (the weights of each neuron, including the bias weight, are stored in the rows):")
    for row in Theta1:
        print("\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")
    print()

    print("Initial Theta2 (the weights of each neuron, including the bias weight, are stored in the rows):")
    for row in Theta2:
        print("\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")
    print("\n")

    print("Training set")
    for i in range(len(X)):
        print(f"\tTraining instance {i+1}")
        print(f"\t\tx: [{X[i,0]:.5f}]")
        print(f"\t\ty: [{y[i,0]:.5f}]")

    print("\n--------------------------------------------")
    print("Computing the error/cost, J, of the network")

    total_cost = 0
    m = X.shape[0]

    a1_list = []
    z2_list = []
    a2_list = []
    z3_list = []
    a3_list = []

    for i in range(m):
        print(f"\tProcessing training instance {i+1}")
        a1, z2, a2, z3, a3 = forward_propagate(X[i], Theta1, Theta2)
        a1_list.append(a1)
        z2_list.append(z2)
        a2_list.append(a2)
        z3_list.append(z3)
        a3_list.append(a3)

        print(f"\tForward propagating the input [{X[i,0]:.5f}]")
        print("\t\ta1: [" + "   ".join(f"{val:.5f}" for val in a1) + "]")
        print()
        print("\t\tz2: [" + "   ".join(f"{val:.5f}" for val in z2) + "]")
        print("\t\ta2: [" + "   ".join(f"{val:.5f}" for val in a2) + "]")
        print()
        print("\t\tz3: [" + "   ".join(f"{val:.5f}" for val in z3) + "]")
        print("\t\ta3: [" + "{:.5f}".format(a3[0]) + "]")
        print()
        print("\t\tf(x): [{:.5f}]".format(a3[0]))
        print("\tPredicted output for instance {}: [{:.5f}]".format(i+1, a3[0]))
        print("\tExpected output for instance {}: [{:.5f}]".format(i+1, y[i,0]))

        cost = -(y[i]*np.log(a3) + (1 - y[i])*np.log(1 - a3))
        print("\tCost, J, associated with instance {}: {:.3f}\n".format(i+1, cost[0]))
        total_cost += cost[0]

    final_cost = total_cost / m
    print(f"Final (regularized) cost, J, based on the complete training set: {final_cost:.5f}\n\n")

    print("\n--------------------------------------------")
    print("Running backpropagation")

    Delta1 = np.zeros_like(Theta1)
    Delta2 = np.zeros_like(Theta2)

    for i in range(m):
        print(f"\tComputing gradients based on training instance {i+1}")
        delta3 = a3_list[i] - y[i]
        delta2 = (Theta2[:,1:].T @ delta3) * sigmoid_gradient(z2_list[i])

        print("\t\tdelta3: [" + "  ".join(f"{val:.5f}" for val in delta3) + "]")
        print("\t\tdelta2: [" + "   ".join(f"{val:.5f}" for val in delta2) + "]")
        print()

        grad2 = delta3[:, None] @ a2_list[i][None, :]
        grad1 = delta2[:, None] @ a1_list[i][None, :]

        print("\t\tGradients of Theta2 based on training instance {}:".format(i+1))
        for row in grad2:
            print("\t\t\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")
        print()

        print("\t\tGradients of Theta1 based on training instance {}:".format(i+1))
        for row in grad1:
            print("\t\t\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")
        print()

        Delta1 += grad1   # ✔ 누적합
        Delta2 += grad2

    Theta1_grad = Delta1 / m   # ✔ 평균
    Theta2_grad = Delta2 / m

    print("\tThe entire training set has been processed. Computing the average (regularized) gradients:")

    print("\t\tFinal regularized gradients of Theta1:")
    for row in Theta1_grad:
        print("\t\t\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")
    print()

    print("\t\tFinal regularized gradients of Theta2:")
    for row in Theta2_grad:
        print("\t\t\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")

if __name__ == "__main__":
    sys.stdout = open("backprop_debug.txt", "w")

    lambda_reg=0
    Theta1 = np.array([[0.4, 0.1], [0.3, 0.2]])
    Theta2 = np.array([[0.7, 0.5, 0.6]])
    X = [0.13, 0.42]
    y = [0.9, 0.23]

    main(Theta1, Theta2, X, y, lambda_reg)

    sys.stdout.close()

