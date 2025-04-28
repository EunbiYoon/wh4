import sys
import numpy as np



def input_vector(X):
    X = np.array(X)
    result = []
    for x in X:
        x = np.array(x).reshape(-1, 1)    # (n,1)로 변환
        x = add_bias(x)                   # 맨 앞에 bias(1) 추가
        result.append(x)
    return result

def add_bias(x):
    # change to numpy array
    x = np.array(x) 
    # add 1 in 0th location in x vector
    return np.insert(x, 0, 1) 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def get_activation(Theta, x):
    z = Theta @ x
    a = sigmoid(z)
    return a

    

def main(Theta1, Theta2, X, y, lambda_reg):
    # 항상 소숫점 5자리로 표시되게 고정
    np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

    # input layer
    x_list = input_vector(X) 
    for i, xi in enumerate(x_list):
        # hidden layer ~ output layer
        a2=get_activation(Theta1, xi)
        print(xi, a2)


if __name__ == "__main__":
    # Regularization parameter
    lambda_reg = 0.250

    # Initial Theta1 (the weights of each neuron, including the bias weight, are stored in the rows):
    Theta1 = [
        [0.42000, 0.15000, 0.40000],
        [0.72000, 0.10000, 0.54000],
        [0.01000, 0.19000, 0.42000],
        [0.30000, 0.35000, 0.68000]
    ]

    # Initial Theta2 (the weights of each neuron, including the bias weight, are stored in the rows):
    Theta2 = [
        [0.21000, 0.67000, 0.14000, 0.96000, 0.87000],
        [0.87000, 0.42000, 0.20000, 0.32000, 0.89000],
        [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]
    ]

    # Initial Theta3 (the weights of each neuron, including the bias weight, are stored in the rows):
    Theta3 = [
        [0.04000, 0.87000, 0.42000, 0.53000],
        [0.17000, 0.10000, 0.95000, 0.69000]
    ]

    # Training set
    # Training instance 1
    X = [
        [0.32000, 0.68000],  # instance 1
        [0.83000, 0.02000]   # instance 2
    ]

    y = [
        [0.75000, 0.98000],  # label for instance 1
        [0.75000, 0.28000]   # label for instance 2
    ]
    main(Theta1, Theta2, X, y, lambda_reg)
