import numpy as np
import debug_text

# ========== Forward Propagation Functions ==========
def add_bias(x):
    x = np.array(x)
    return np.insert(x, 0, 1, axis=0)

def input_vector(X):
    X = np.array(X)
    result = []
    for x in X:
        x = np.array(x).reshape(-1, 1)  # (n,1)로 변환
        x = add_bias(x)                 # 맨 앞에 bias(1) 추가
        result.append(x)
    return result

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def get_activation(Theta_i, x_i):
    Theta_i = np.array(Theta_i)
    z = Theta_i @ x_i
    a = sigmoid(z)
    return z, a

def forward_propagation(Theta, X):
    x_list = input_vector(X)
    all_a_lists = []  # 모든 instance의 레이어별 a값 저장 리스트
    all_z_lists = []  # 모든 instance의 레이어별 z값 저장 리스트

    for i, x_i in enumerate(x_list):
        a_list = []  # 하나의 인스턴스에 대해 레이어별 a값 저장
        z_list = []  # 하나의 인스턴스에 대해 레이어별 z값 저장

        a_list.append(x_i)  # input layer (bias 포함)

        for layer_idx, Theta_i in enumerate(Theta):
            z, a = get_activation(Theta_i, x_i)
            z_list.append(z)

            if layer_idx != len(Theta) - 1:
                a = add_bias(a)

            x_i = a
            a_list.append(a)

        all_a_lists.append(a_list)
        all_z_lists.append(z_list)

    return all_a_lists, all_z_lists

# ========== Backward Propagation Functions ==========
def log_func(x):
    return np.log(x)

def cost_function(pred_y_list, true_y_list, Theta, lambda_reg):
    J_list = []

    for pred_y, true_y in zip(pred_y_list, true_y_list):
        pred_y = np.array(pred_y).reshape(-1, 1)
        true_y = np.array(true_y).reshape(-1, 1)
        # (행렬 버전: J = - [ yᵗ log(f) + (1 - y)ᵗ log(1 - f) ])
        cost = -(true_y.T @ np.log(pred_y) + (1 - true_y).T @ np.log(1 - pred_y))
        J_list.append(cost.item())

    m = len(true_y_list)
    # J = J/n
    cost_no_reg = sum(J_list) / m

    # ✅ 정규화 항 추가
    reg_term = 0
    for theta in Theta:
        theta = np.array(theta)
        # S = theta^2
        reg_term += np.sum(theta[:, 1:] ** 2)
    # S = S * (lambda / 2m)
    reg_term = reg_term* (lambda_reg / (2 * m))

    # J_list, final_J = J+S
    return J_list, cost_no_reg + reg_term

def blame_delta(Theta, a_list, y):
    Theta = [np.array(theta_i) for theta_i in Theta]
    delta_list = [None] * len(Theta)

    for layer_idx in reversed(range(len(Theta))):
        Theta_i = Theta[layer_idx]

        if layer_idx == len(Theta) - 1:
            # 출력층
            delta = a_list[layer_idx+1] - np.array(y).reshape(-1, 1)
            delta_list[layer_idx] = delta
        else:
            delta_next = delta_list[layer_idx + 1]
            Theta_next = Theta[layer_idx + 1]

            delta = (Theta_next[:,1:].T @ delta_next) * (a_list[layer_idx+1][1:] * (1 - a_list[layer_idx+1][1:]))
            delta_list[layer_idx] = delta
    return delta_list

def gradient_theta(delta_list, a_list):
    D_list = []

    # 🔥 순방향으로 순회 (reversed 제거)
    for i in range(len(delta_list)):
        delta = delta_list[i]
        a = a_list[i]

        grad = delta @ a.T
        D_list.append(grad)
    return D_list

def regularized_gradient_theta(all_D_lists, Theta, lambda_reg, m):
    Theta = [np.array(theta_i) for theta_i in Theta]

    num_layers = len(Theta)
    avg_D_list = []

    for layer_idx in range(num_layers):
        # ⬇ 해당 레이어에 대해 모든 인스턴스의 gradient 누적 합산
        sum_grad = sum(instance_D[layer_idx] for instance_D in all_D_lists)

        # ⬇ 평균 계산
        avg_grad = sum_grad / m

        # ⬇ 정규화 (bias 제외하고)
        avg_grad[:, 1:] += (lambda_reg / m) * Theta[layer_idx][:, 1:]

        avg_D_list.append(avg_grad)

    return avg_D_list

def backpropagation(Theta, all_a_lists, y, lambda_reg):
    # ✅ 평균 누적용 초기화
    all_D_lists = []
    all_delta_lists = []
    average_D_lists=[]

    for i, a_list in enumerate(all_a_lists):
        # a 만큼 detla 값 반복해서 쌓기
        delta_list = blame_delta(Theta, a_list, y[i])
        all_delta_lists.append(delta_list)
        # a 만큼 D 값 반복해서 쌓기
        D_list = gradient_theta(delta_list, a_list)
        all_D_lists.append(D_list)
    
    # ✅ 정규화된 gradient 계산
    m = len(all_a_lists)  # training instance #
    finalized_D = regularized_gradient_theta(all_D_lists, Theta, lambda_reg, m)
    return finalized_D, all_D_lists, all_delta_lists

# ========== Main ==========
def main(Theta, X, y, lambda_reg):
    np.set_printoptions(precision=5, suppress=True, floatmode='fixed')
    # forward propagation 
    all_a_lists, all_z_lists = forward_propagation(Theta, X)

    # cost function 
    pred_y_list = [a_list[-1] for a_list in all_a_lists]
    true_y_list = y
    J_list, final_cost = cost_function(pred_y_list, true_y_list, Theta, lambda_reg)
    finalized_D, D_list, delta_list = backpropagation(Theta, all_a_lists, y, lambda_reg)
    debug_text.main(lambda_reg, X, y, Theta, all_a_lists, all_z_lists, J_list, final_cost, delta_list, D_list, finalized_D)


# ========== Entry Point ==========
if __name__ == "__main__":
    ######### Example 1
    # lambda_reg = 0
    # Theta = [
    #     [[0.4, 0.1], [0.3, 0.2]],
    #     [[0.7, 0.5, 0.6]]
    # ]
    # X = [[0.13], [0.42]]
    # y = [[0.9], [0.23]]

    # ########## Example 2
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