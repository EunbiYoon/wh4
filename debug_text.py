import numpy as np

def main(lambda_reg, X, y, Theta, all_a_lists, all_z_lists, J_list, final_cost, delta_list, D_list, finalized_D, DEBUG_FILENAME, header=""): 
    # Open a debug file for writing results
    with open(f"debug/backprop_{DEBUG_FILENAME}.txt", "w", encoding="utf-8") as f:
        if header:
            # Write optional header if provided
            f.write("=" * 80 + "\n")
            f.write(header + "\n")
            f.write("=" * 80 + "\n")
        
        # Set NumPy printing options for consistent formatting
        np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

        # Helper function to write a line with newline
        def println(line=""):
            f.write(line + "\n")

        # Print regularization parameter
        println(f"Regularization parameter lambda={lambda_reg:.3f}\n")

        # Print network structure from input layer through each Theta layer
        structure = [len(X[0])] + [len(t) for t in Theta]
        println("Initializing the network with the following structure (number of neurons per layer): " + str(structure).replace(",", "") + "\n")

        # Print initial weights (Theta) for each layer
        for i, theta in enumerate(Theta):
            println(f"Initial Theta{i+1} (the weights of each neuron, including the bias weight, are stored in the rows):")
            for row in theta:
                println("\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")
            println()

        # Print the training data (X, y) for each instance
        println("\nTraining set")
        for i, (x_i, y_i) in enumerate(zip(X, y)):
            println(f"\tTraining instance {i+1}")
            println(f"\t\tx: [{ '   '.join(f'{val:.5f}' for val in x_i) }]")
            println(f"\t\ty: [{ '   '.join(f'{val:.5f}' for val in y_i) }]")

        # Print cost J computed per instance
        println("\n--------------------------------------------")    
        println("Computing the error/cost, J, of the network")

        # Loop through each training instance to print forward propagation results
        for i, (x_i, y_i, a_list, z_list, cost) in enumerate(zip(X, y, all_a_lists, all_z_lists, J_list)):
            println(f"\tProcessing training instance {i+1}")
            println(f"\tForward propagating the input [{ '   '.join(f'{val:.5f}' for val in x_i) }]")
            
            # Print input layer activation (a1)
            println(f"\t\ta1: [{ '   '.join(f'{val.item():.5f}' for val in a_list[0]) }]\n")

            # Print z and a values for each hidden/output layer
            for layer_idx in range(len(z_list)):
                z_num = layer_idx + 2
                a_num = layer_idx + 2
                println(f"\t\tz{z_num}: [{ '   '.join(f'{val.item():.5f}' for val in z_list[layer_idx]) }]")
                println(f"\t\ta{a_num}: [{ '   '.join(f'{val.item():.5f}' for val in a_list[layer_idx + 1]) }]\n")

            # Print predicted and expected output, and cost J for the instance
            fx = a_list[-1]
            println(f"\t\tf(x): [{ '   '.join(f'{val.item():.5f}' for val in fx) }]")
            println(f"\tPredicted output for instance {i+1}: [{ '   '.join(f'{val.item():.5f}' for val in fx) }]")
            println(f"\tExpected output for instance {i+1}: [{ '   '.join(f'{val:.5f}' for val in y_i) }]")
            println(f"\tCost, J, associated with instance {i+1}: {cost:.3f}\n")

        # Print final cost J over the full training set
        println(f"Final (regularized) cost, J, based on the complete training set: {final_cost:.5f}")

        # Begin printing backpropagation results
        println("\n\n\n--------------------------------------------")
        println("Running backpropagation")

        # Loop through each training instance to show delta and gradient per layer
        for i, (delta_set, grad_set) in enumerate(zip(delta_list, D_list)):
            println(f"\tComputing gradients based on training instance {i+1}")
            
            # Print deltas for each layer in reverse order
            for j, delta in enumerate(delta_set[::-1]):
                layer_num = len(delta_set) + 1 - j
                flat = [f"{v.item():.5f}" for v in delta.flatten()]
                println(f"\t\tdelta{layer_num}: [{ '   '.join(flat) }]")
            println()

            # Print unregularized gradient matrices for each layer
            for j, grad in enumerate(grad_set[::-1]):
                theta_num = len(grad_set) - j
                println(f"\t\tGradients of Theta{theta_num} based on training instance {i+1}:")
                for row in grad:
                    println("\t\t\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")
                println()

        # Print final regularized gradients after averaging over all training instances
        println("\tThe entire training set has been processed. Computing the average (regularized) gradients:")
        for i, grad in enumerate(finalized_D):
            println(f"\t\tFinal regularized gradients of Theta{i+1}:")
            for row in grad:
                println("\t\t\t" + "  ".join(f"{val:.5f}" for val in row) + "  ")
            println()
