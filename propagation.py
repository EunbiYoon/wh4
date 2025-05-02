# === Vectorized Neural Network Implementation ===
import numpy as np
import debug_text as debug_text

DEBUG_FILENAME="example"

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def add_bias(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

def forward_propagation(Theta, X):
    Theta = [np.array(t) for t in Theta]
    A = [add_bias(np.array(X))] 
    Z = []

    for i, Theta_i in enumerate(Theta):
        Z_i = A[-1] @ Theta_i.T
        A_i = sigmoid(Z_i)
        Z.append(Z_i)
        if i < len(Theta) - 1:
            A_i = add_bias(A_i) # add bias in hidden layer only
        A.append(A_i)

    # Convert to per-instance format for compatibility
    all_a_lists = [[a[i].reshape(-1, 1) for a in A] for i in range(X.shape[0])]
    all_z_lists = [[z[i].reshape(-1, 1) for z in Z] for i in range(X.shape[0])]

    return A, Z, all_a_lists, all_z_lists

def cost_function(A_final, Y, Theta, lambda_reg):
    m = Y.shape[0]
    cost = -np.sum(Y * np.log(A_final) + (1 - Y) * np.log(1 - A_final)) / m

    reg_term = 0
    for theta in Theta:
        theta = np.array(theta)
        # remove bias -> [:, 1:]
        reg_term += np.sum(theta[:, 1:] ** 2) 
    reg_term = reg_term * (lambda_reg / (2 * m))

    return cost, cost + reg_term

def backpropagation_vectorized(Theta, A, Z, Y, lambda_reg):
    Theta = [np.array(t) for t in Theta]
    m = Y.shape[0]
    delta = A[-1] - Y
    gradients = [None] * len(Theta)

    for i in reversed(range(len(Theta))):
        a_prev = A[i]
        gradients[i] = (delta.T @ a_prev) / m
        if i > 0:
            delta = (delta @ Theta[i][:, 1:]) * sigmoid_gradient(Z[i - 1])

    for i in range(len(Theta)):
        # remove bias -> [:, 1:]
        gradients[i][:, 1:] += (lambda_reg / m) * Theta[i][:, 1:]

    return gradients

def run_debug(Theta, X, y, lambda_reg, DEBUG_FILENAME):
    np.set_printoptions(precision=5, suppress=True, floatmode='fixed')
    A, Z, all_a_lists, all_z_lists = forward_propagation(Theta, X)
    pred_y_list = [a_list[-1] for a_list in all_a_lists]
    true_y_list = [y[i].reshape(-1, 1) for i in range(y.shape[0])]

    J_list = []
    for pred, true in zip(pred_y_list, true_y_list):
        J = -(true.T @ np.log(pred) + (1 - true).T @ np.log(1 - pred))
        J_list.append(J.item())

    delta_list = []
    D_list = []
    for i in range(X.shape[0]):
        delta_i = [None] * len(Theta)
        delta_i[-1] = all_a_lists[i][-1] - y[i].reshape(-1, 1)
        for l in reversed(range(len(Theta) - 1)):
            delta_i[l] = (Theta[l + 1][:, 1:].T @ delta_i[l + 1]) * sigmoid_gradient(all_z_lists[i][l])
        delta_list.append(delta_i)

        D_i = []
        for l in range(len(Theta)):
            D_i.append(delta_i[l] @ all_a_lists[i][l].T)
        D_list.append(D_i)

    finalized_D = backpropagation_vectorized(Theta, A, Z, y, lambda_reg)
    _, final_cost = cost_function(A[-1], y, Theta, lambda_reg)

    debug_text.main(lambda_reg, X, y, Theta, all_a_lists, all_z_lists, J_list, final_cost, delta_list, D_list, finalized_D, DEBUG_FILENAME)

if __name__ == "__main__":
    ### Example 1
    # lambda_reg = 0
    # Theta = [
    #     [[0.4, 0.1], [0.3, 0.2]],
    #     [[0.7, 0.5, 0.6]]
    # ]
    # X = [[0.13], [0.42]]
    # y = [[0.9], [0.23]]

    ### Example 2
    lambda_reg = 0.250
    X = [
        [0.32000, 0.68000],
        [0.83000, 0.02000]
    ]
    y = [
        [0.75000, 0.98000],
        [0.75000, 0.28000]
    ]
    Theta = [
        [
            [0.42000, 0.15000, 0.40000],
            [0.72000, 0.10000, 0.54000],
            [0.01000, 0.19000, 0.42000],
            [0.30000, 0.35000, 0.68000]
        ],
        [
            [0.21000, 0.67000, 0.14000, 0.96000, 0.87000],
            [0.87000, 0.42000, 0.20000, 0.32000, 0.89000],
            [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]
        ],
        [
            [0.04000, 0.87000, 0.42000, 0.53000],
            [0.17000, 0.10000, 0.95000, 0.69000]
        ]
    ]

    ### change form 
    Theta = [np.array(t) for t in Theta]
    X = np.array(X)
    y = np.array(y)
    run_debug(Theta, X, y, lambda_reg, DEBUG_FILENAME)
