import numpy as np

# forward propagation
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

def forward_propagation(Theta, X):
     # input layer
    x_list = input_vector(X) 
    output=0
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
                output=a
            
            # previous activation become next input
            x_i=a
    return output

# backward propagation
def log_func(x):
    return np.log10(x)

def cost_function(pred_y, true_y):
    pred_y = np.array(pred_y).reshape(-1, 1)  # ✅ 리스트 → np.array로 변환
    true_y = np.array(true_y).reshape(-1, 1)  # ✅ 리스트 → np.array로 변환

    m = true_y.shape[0]
    cost = -(1/m) * (true_y.T @ np.log(pred_y) + (1 - true_y).T @ np.log(1 - pred_y))
    return cost.squeeze()

def blame_delta(Theta, X, y):
    return Theta.T*previos

def main(Theta, X, y, lambda_reg):
    # Always show 5 decimal places
    np.set_printoptions(precision=5, suppress=True, floatmode='fixed')

    # Forward propagation -> predicted output
    predicted_activation=forward_propagation(Theta, X)
    predicted_activation=predicted_activation.reshape(-1)
    
    # Backward propagation -> 
    cost_list=[]
    for i in range(len(X)):
        J=cost_function(predicted_activation, y[i])
        print(f"J:{J}")

        

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
