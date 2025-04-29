import numpy as np
import sys

def debug_text(lambda_reg, Theta, X, y, all_activations, all_deltas, all_grads, final_cost, final_grads):
    with open("back_prop_example.txt", "w") as f:
        # Header
        f.write(f"Regularization parameter lambda={lambda_reg:.3f}\n\n")
        f.write("Initializing the network with the following structure (number of neurons per layer): ")
        structure = [len(X[0])] + [len(t) for t in Theta]
        f.write(str(structure).replace(",", "") + "\n\n")

        for i, theta in enumerate(Theta):
            f.write(f"Initial Theta{i+1} (the weights of each neuron, including the bias weight, are stored in the rows):\n")
            for row in theta:
                f.write("\t" + "  ".join(f"{val: .5f}" for val in row) + "  \n")
            f.write("\n")

        # Training Set
        f.write("Training set\n")
        for i, (x_i, y_i) in enumerate(zip(X, y)):
            f.write(f"\tTraining instance {i+1}\n")
            f.write(f"\t\tx: [{ '   '.join(f'{val:.5f}' for val in x_i) }]\n")
            f.write(f"\t\ty: [{ '   '.join(f'{val:.5f}' for val in y_i) }]\n")
        f.write("\n--------------------------------------------\n")
        f.write("Computing the error/cost, J, of the network\n")

        for i, act in enumerate(all_activations):
            a1, z2, a2, z3, a3, z4, a4 = act
            f.write(f"\tProcessing training instance {i+1}\n")
            f.write(f"\tForward propagating the input [{ '   '.join(f'{v:.5f}' for v in X[i]) }]\n")
            f.write(f"\t\ta1: [{ '   '.join(f'{v:.5f}' for v in a1) }]\n\n")
            f.write(f"\t\tz2: [{ '   '.join(f'{v:.5f}' for v in z2) }]\n")
            f.write(f"\t\ta2: [{ '   '.join(f'{v:.5f}' for v in a2) }]\n\n")
            f.write(f"\t\tz3: [{ '   '.join(f'{v:.5f}' for v in z3) }]\n")
            f.write(f"\t\ta3: [{ '   '.join(f'{v:.5f}' for v in a3) }]\n\n")
            f.write(f"\t\tz4: [{ '   '.join(f'{v:.5f}' for v in z4) }]\n")
            f.write(f"\t\ta4: [{ '   '.join(f'{v:.5f}' for v in a4) }]\n\n")
            f.write(f"\t\tf(x): [{ '   '.join(f'{v:.5f}' for v in a4) }]\n")
            f.write(f"\tPredicted output for instance {i+1}: [{ '   '.join(f'{v:.5f}' for v in a4) }]\n")
            f.write(f"\tExpected output for instance {i+1}: [{ '   '.join(f'{v:.5f}' for v in y[i]) }]\n")
            cost_i = -(np.array(y[i]).T @ np.log(a4) + (1 - np.array(y[i])).T @ np.log(1 - a4))
            f.write(f"\tCost, J, associated with instance {i+1}: {cost_i:.3f}\n\n")

        f.write(f"Final (regularized) cost, J, based on the complete training set: {final_cost:.5f}\n\n\n")
        f.write("--------------------------------------------\n")
        f.write("Running backpropagation\n")

        for i, (deltas, grads) in enumerate(zip(all_deltas, all_grads)):
            f.write(f"\tComputing gradients based on training instance {i+1}\n")
            for j, delta in enumerate(deltas[::-1]):
                f.write(f"\t\tdelta{len(deltas)+1-j}: [{ '   '.join(f'{val:.5f}' for val in delta.flatten()) }]\n")

            for j, grad in enumerate(grads[::-1]):
                f.write(f"\n\t\tGradients of Theta{len(grads)-j} based on training instance {i+1}:\n")
                for row in grad:
                    f.write("\t\t\t" + "  ".join(f"{val:.5f}" for val in row) + "  \n")
            f.write("\n")

        f.write("\tThe entire training set has been processed. Computing the average (regularized) gradients:\n")
        for i, final_grad in enumerate(final_grads):
            f.write(f"\n\t\tFinal regularized gradients of Theta{i+1}:\n")
            for row in final_grad:
                f.write("\t\t\t" + "  ".join(f"{val:.5f}" for val in row) + "  \n")
