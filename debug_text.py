import numpy as np

def main(lambda_reg, X, y, Theta, all_a_lists, all_z_lists, J_list, final_cost, delta_list, D_list, finalized_D): 
    with open("backprop_debug.txt", "w", encoding="utf-8") as f:
        np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

        def println(line=""):
            f.write(line + "\n")

        println(f"Regularization parameter lambda={lambda_reg:.3f}\n")

        structure = [len(X[0])] + [len(t) for t in Theta]
        println("Initializing the network with the following structure (number of neurons per layer): " + str(structure).replace(",", "") + "\n")

        for i, theta in enumerate(Theta):
            println(f"Initial Theta{i+1} (the weights of each neuron, including the bias weight, are stored in the rows):")
            for row in theta:
                println("\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")
            println()

        println("\nTraining set")
        for i, (x_i, y_i) in enumerate(zip(X, y)):
            println(f"\tTraining instance {i+1}")
            println(f"\t\tx: [{ '   '.join(f'{val:.5f}' for val in x_i) }]")
            println(f"\t\ty: [{ '   '.join(f'{val:.5f}' for val in y_i) }]")

        println("\n--------------------------------------------")    
        println("Computing the error/cost, J, of the network")

        for i, (x_i, y_i, a_list, z_list, cost) in enumerate(zip(X, y, all_a_lists, all_z_lists, J_list)):
            println(f"\tProcessing training instance {i+1}")
            println(f"\tForward propagating the input [{ '   '.join(f'{val:.5f}' for val in x_i) }]")
            
            # a1 (input with bias)
            println(f"\t\ta1: [{ '   '.join(f'{val.item():.5f}' for val in a_list[0]) }]\n")

            # 출력 z2~zL, a2~aL
            for layer_idx in range(len(z_list)):
                z_num = layer_idx + 2
                a_num = layer_idx + 2
                println(f"\t\tz{z_num}: [{ '   '.join(f'{val.item():.5f}' for val in z_list[layer_idx]) }]")
                println(f"\t\ta{a_num}: [{ '   '.join(f'{val.item():.5f}' for val in a_list[layer_idx + 1]) }]\n")

            # 최종 출력
            fx = a_list[-1]
            println(f"\t\tf(x): [{ '   '.join(f'{val.item():.5f}' for val in fx) }]")
            println(f"\tPredicted output for instance {i+1}: [{ '   '.join(f'{val.item():.5f}' for val in fx) }]")
            println(f"\tExpected output for instance {i+1}: [{ '   '.join(f'{val:.5f}' for val in y_i) }]")
            println(f"\tCost, J, associated with instance {i+1}: {cost:.3f}\n")

        println(f"Final (regularized) cost, J, based on the complete training set: {final_cost:.5f}")

        println("\n\n\n--------------------------------------------")
        println("Running backpropagation")

        for i, (delta_set, grad_set) in enumerate(zip(delta_list, D_list)):
            println(f"\tComputing gradients based on training instance {i+1}")
            for j, delta in enumerate(delta_set[::-1]):
                layer_num = len(delta_set) + 1 - j
                flat = [f"{v.item():.5f}" for v in delta.flatten()]
                println(f"\t\tdelta{layer_num}: [{ '   '.join(flat) }]")
            println()
            for j, grad in enumerate(grad_set[::-1]):
                theta_num = len(grad_set) - j
                println(f"\t\tGradients of Theta{theta_num} based on training instance {i+1}:")
                for row in grad:
                    println("\t\t\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")
                println()

        println("\tThe entire training set has been processed. Computing the average (regularized) gradients:")
        for i, grad in enumerate(finalized_D):
            println(f"\t\tFinal regularized gradients of Theta{i+1}:")
            for row in grad:
                println("\t\t\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")
            println()
