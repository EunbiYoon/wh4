# === numerical_gradients.py ===
import numpy as np
from propagation import forward_propagation, cost_function
import debug_text

DATASET_NAME = "extra"

def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])

def unflatten_weights(flattened, shapes):
    weights = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        weights.append(flattened[idx:idx + size].reshape(shape))
        idx += size
    return weights

def compute_numerical_gradient(J, theta, epsilon):
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon
        grad[i] = (J(theta_plus) - J(theta_minus)) / (2 * epsilon)
    return grad

def make_cost_function(X, y, shapes, lambda_reg):
    def J(theta_flat):
        weights = unflatten_weights(theta_flat, shapes)
        A, *_ = forward_propagation(weights, X)
        return cost_function(A[-1], y, weights, lambda_reg)[1]
    return J

def numerical_gradient_only(Theta, X, y, lambda_reg, epsilons):
    shapes = [w.shape for w in Theta]
    theta_flat = flatten_weights(Theta)
    J = make_cost_function(X, y, shapes, lambda_reg)

    for eps in epsilons:
        print(f"\n=== Numerical Gradient with epsilon = {eps} ===")
        numerical_grad = compute_numerical_gradient(J, theta_flat, eps)
        for i, g in enumerate(numerical_grad):
            print(f"theta[{i}] => Numerical Gradient: {g:.8f}")

if __name__ == "__main__":
    DEBUG_FILENAME = "example1_debug"

    ### Example 1
    lambda_reg = 0
    Theta = [
        [[0.4, 0.1], [0.3, 0.2]],
        [[0.7, 0.5, 0.6]]
    ]
    X = [[0.13], [0.42]]
    y = [[0.9], [0.23]]

    Theta = [np.array(t) for t in Theta]
    X = np.array(X)
    y = np.array(y)

    # Run numerical gradient only
    numerical_gradient_only(Theta, X, y, lambda_reg, epsilons=[0.1, 1e-6])
    ### change form 
    Theta = [np.array(t) for t in Theta]
    X = np.array(X)
    y = np.array(y)
    
