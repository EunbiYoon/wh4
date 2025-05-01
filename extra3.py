import numpy as np

EPS=[0.1, 0.000001]

# === Activation functions ===
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    s = sigmoid(z)
    return s * (1 - s)

# === Helpers to flatten/unflatten weight matrices ===
def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])

def unflatten_weights(flat, shapes):
    weights = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        weights.append(flat[idx:idx + size].reshape(shape))
        idx += size
    return weights

# === Forward propagation ===
def forward_propagation(Theta, X):
    A = [X]
    Z = []
    for theta in Theta:
        X = np.insert(X, 0, 1, axis=1)  # add bias
        z = X @ theta.T
        Z.append(z)
        X = sigmoid(z)
        A.append(X)
    return A, Z

# === Cost function ===
def cost_function(A_last, Y, Theta, lambda_reg):
    m = Y.shape[0]
    epsilon = 1e-8  # avoid log(0)
    J = -np.sum(Y * np.log(A_last + epsilon) + (1 - Y) * np.log(1 - A_last + epsilon)) / m

    # Regularization (if needed)
    reg = sum(np.sum(theta[:, 1:] ** 2) for theta in Theta)
    J += (lambda_reg / (2 * m)) * reg
    return J

# === Cost function wrapper ===
def make_cost_function(X, y, shapes, lambda_reg):
    def J(theta_flat):
        Theta = unflatten_weights(theta_flat, shapes)
        A, _ = forward_propagation(Theta, X)
        return cost_function(A[-1], y, Theta, lambda_reg)
    return J

# === Numerical gradient computation ===
def compute_numerical_gradient(J, theta_flat, epsilon):
    num_grad = np.zeros_like(theta_flat)
    for i in range(len(theta_flat)):
        theta_plus = theta_flat.copy()
        theta_minus = theta_flat.copy()
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon
        num_grad[i] = (J(theta_plus) - J(theta_minus)) / (2 * epsilon)
    return num_grad

# === Main execution for fixed example ===
if __name__ == "__main__":
    # Input from example
    X = np.array([[0.13], [0.42]])  # shape (2, 1)
    y = np.array([[0.9], [0.23]])   # shape (2, 1)

    # Fixed weights from example1
    Theta = [
        np.array([[0.4, 0.1], [0.3, 0.2]]),       # Theta1: 2 neurons, 1 input + bias
        np.array([[0.7, 0.5, 0.6]])              # Theta2: 1 neuron, 2 inputs + bias
    ]

    for i in range(len(EPS)):
        lambda_reg = 0  # no regularization
        shapes = [t.shape for t in Theta]
        flat_Theta = flatten_weights(Theta)

        # Compute cost function wrapper
        cost_func = make_cost_function(X, y, shapes, lambda_reg)

        # Compute numerical gradient
        numerical_grad = compute_numerical_gradient(cost_func, flat_Theta, EPS[i])

        # Convert back to matrix form
        numerical_grad_matrices = unflatten_weights(numerical_grad, shapes)

        # Print results
        for j, grad in enumerate(numerical_grad_matrices):
            print(f"\nNumerical Gradient for Theta{j+1} When epsilon = {EPS[i]}:")
            for row in grad:
                print("  ".join(f"{val:.5f}" for val in row))
