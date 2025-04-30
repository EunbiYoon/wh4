import numpy as np
import sys



def main(lambda_reg, X, y, Theta, all_a_lists, all_z_lists, J_list, final_cost, delta_list, D_list, finalized_D): 
    sys.stdout = open("backprop_debug.txt", "w")

    # ✅ 정식 함수 이름으로 수정
    np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

    print(f"Regularization parameter lambda={lambda_reg:.3f}\n")

    # 구조 출력
    structure = [len(X[0])] + [len(t) for t in Theta]
    print("Initializing the network with the following structure (number of neurons per layer):", str(structure).replace(",", ""), "\n")

    # Theta 출력
    for i, theta in enumerate(Theta):
        print(f"Initial Theta{i+1} (the weights of each neuron, including the bias weight, are stored in the rows):")
        for row in theta:
            print("\t" + "  ".join(f"{val:.5f}" for val in row))
        print()

    # Training set 출력
    print("\nTraining set")
    for i, (x_i, y_i) in enumerate(zip(X, y)):
        print(f"\tTraining instance {i+1}")
        print(f"\t\tx: [{ '   '.join(f'{val:.5f}' for val in x_i) }]")
        print(f"\t\ty: [{ '   '.join(f'{val:.5f}' for val in y_i) }]")

    print("\n--------------------------------------------")    
    print("Computing the error/cost, J, of the network")
    for i, (x_i, y_i, a_list, z_list, cost) in enumerate(zip(X, y, all_a_lists, all_z_lists, J_list)):
        print(f"\tProcessing training instance {i+1}")
        print(f"\tForward propagating the input [{ '   '.join(f'{val:.5f}' for val in x_i) }]")
        
        # a1 (input with bias)
        print(f"\t\ta1: [{ '   '.join(f'{val.item():.5f}' for val in a_list[0]) }]\n")

        # z2, a2
        print(f"\t\tz2: [{ '   '.join(f'{val.item():.5f}' for val in z_list[0]) }]")
        print(f"\t\ta2: [{ '   '.join(f'{val.item():.5f}' for val in a_list[1]) }]\n")

        # z3, a3
        print(f"\t\tz3: [{ '   '.join(f'{val.item():.5f}' for val in z_list[1]) }]")
        print(f"\t\ta3: [{ '   '.join(f'{val.item():.5f}' for val in a_list[2]) }]\n")

        # f(x)
        fx = a_list[-1]
        print(f"\t\tf(x): [{ '   '.join(f'{val.item():.5f}' for val in fx) }]")
        print(f"\tPredicted output for instance {i+1}: [{ '   '.join(f'{val.item():.5f}' for val in fx) }]")
        print(f"\tExpected output for instance {i+1}: [{ '   '.join(f'{val:.5f}' for val in y_i) }]")
        print(f"\tCost, J, associated with instance {i+1}: {cost:.3f}\n")

    print(f"Final (regularized) cost, J, based on the complete training set: {final_cost:.5f}")

    print("\n\n\n--------------------------------------------")
    print("Running backpropagation")

    for i, (delta_list, grad_list) in enumerate(zip(delta_list, D_list)):
        print(f"\tComputing gradients based on training instance {i+1}")
        
        # delta 출력
        for j, delta in enumerate(delta_list[::-1]):
            layer_num = len(delta_list) + 1 - j
            flat_delta = [f"{val.item():.5f}" for val in delta.flatten()]
            print(f"\t\tdelta{layer_num}: [{ '   '.join(flat_delta) }]")
        print()

        # Gradients 출력
        for j, grad in enumerate(grad_list[::-1]):
            theta_num = len(grad_list) - j
            print(f"\t\tGradients of Theta{theta_num} based on training instance {i+1}:")
            for row in grad:
                try:
                    print("\t\t\t" + "  ".join(f"{val:.5f}" for val in row))
                except TypeError:
                    print("\t\t\t" + f"{row:.5f}")
            print()

    print("\tThe entire training set has been processed. Computing the average (regularized) gradients:")
    for i, grad in enumerate(finalized_D):
        print(f"\t\tFinal regularized gradients of Theta{i+1}:")
        for row in grad:
            try:
                print("\t\t\t" + "  ".join(f"{val:.5f}" for val in row))
            except TypeError:
                print("\t\t\t" + f"{row:.5f}")
        print()

    sys.stdout.close()