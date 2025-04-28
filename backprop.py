import numpy as np

def add_bias(x):
    """Add bias term (1) at the top of the vector"""
    x = np.array(x)
    return np.insert(x, 0, 1, axis=0)

def input_vector(X):
    """Prepare input vectors, add bias (No print here)"""
    X = np.array(X)
    result = []
    for x in X:
        x = np.array(x).reshape(-1, 1)  # (n,1)로 변환
        x = add_bias(x)                 # 맨 앞에 bias(1) 추가
        result.append(x)
    return result

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def get_activation(Theta_i, x_i):
    Theta_i = np.array(Theta_i)
    z = Theta_i @ x_i
    a = sigmoid(z)
    return z,a

def main(Theta, X, y, lambda_reg):
    # Always show 5 decimal places
    np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

    # input layer
    x_list = input_vector(X) 

    for i, x_i in enumerate(x_list):
        print(f"=== Training instance {i+1} ===")
        print(f"Input Layer : {x_i.reshape(-1)}\n")

        for layer_idx, Theta_i in enumerate(Theta):
            z,a = get_activation(Theta_i, x_i)
            
            # 마지막 레이어가 아니면 bias 추가
            if layer_idx != len(Theta) - 1:
                a = add_bias(a)
                # Hidden layer 출력
                print(f"Hidden Layer {layer_idx+1} z: {z.reshape(-1)}")
                print(f"Hidden Layer {layer_idx+1} a: {a.reshape(-1)}\n")
            else:
                # Output layer 출력
                print(f"Output Layer z: {z.reshape(-1)}")
                print(f"Output Layer a: {a.reshape(-1)}\n")
            
            # previous activation become next input
            x_i=a
        
        print()  # Training instance 구분


if __name__ == "__main__":
    ########## Example 1
    # lambda_reg = 0
    # Theta = [
    #     [[0.4, 0.1], [0.3, 0.2]], 
    #     [[0.7, 0.5, 0.6]]
    # ]
    # X = [0.13, 0.42]
    # y = [0.9, 0.23]


    ########## Example 2
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
    
    main(Theta, X, y, lambda_reg)
