# === numerical_gradients.py ===
import numpy as np
from propagation import forward_propagation, cost_function
from nn import NeuralNetwork, load_dataset, stratified_k_fold_split, my_accuracy, my_f1_score, save_metrics_table, plot_best_learning_curve

DATASET_NAME = "wdbc"

# === Flatten and Unflatten Functions ===
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

# === Cost Function Wrapper ===
def make_cost_function(X, y, shapes, lam):
    def J(theta_flat):
        weights = unflatten_weights(theta_flat, shapes)
        all_a_lists, _ = forward_propagation(weights, X)
        pred_y_list = [a_list[-1] for a_list in all_a_lists]
        _, final_cost = cost_function(pred_y_list, y, weights, lam)
        return final_cost
    return J

# === Numerical Gradient Computation ===
def compute_numerical_gradient(J, theta, epsilon=1e-4):
    num_grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon
        num_grad[i] = (J(theta_plus) - J(theta_minus)) / (2 * epsilon)
    return num_grad

# === Main Entry for Numerical Gradient Update ===
def numerical_gradient_update(weights, X, y, lam, alpha):
    shapes = [w.shape for w in weights]
    theta_flat = flatten_weights(weights)
    J = make_cost_function(X, y, shapes, lam)
    grad_flat = compute_numerical_gradient(J, theta_flat)
    grad_structured = unflatten_weights(grad_flat, shapes)
    # Update weights manually here and return new ones
    updated_weights = [w - alpha * g for w, g in zip(weights, grad_structured)]
    return updated_weights


def main():
    X, y = load_dataset()
    folds = stratified_k_fold_split(X, y, k=5)

    lam = [0.25, 0.5]
    hidden_layers = [[8, 6], [8, 6, 4], [4]]
    
    alpha = 0.01
    epochs = 100
    batch_size = 16         # 🔹 원하는 배치 사이즈
    mode = "mini-batch"     # 🔹 또는 "batch"

    dataset_name = DATASET_NAME
    results = {}

    for h_idx, hidden in enumerate(hidden_layers):
        for l_idx, l in enumerate(lam):
            for i, (train_df, test_df) in enumerate(folds):
                X_train = train_df.drop(columns=['label']).values
                y_train = train_df['label'].values.reshape(-1, 1)
                X_test = test_df.drop(columns=['label']).values
                y_test = test_df['label'].values.astype(int).ravel()

                model = NeuralNetwork(
                    layer_sizes=[X_train.shape[1], *hidden, 1],
                    alpha=alpha,
                    lam=l
                )

                # 🔽 mode, batch_size 함께 전달
                model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    fold_index=i,
                    mode=mode
                )

                preds = model.predict(X_test)
                preds_binary = (preds >= 0.5).astype(int).ravel()

                acc = my_accuracy(y_test, preds_binary)
                f1 = my_f1_score(y_test, preds_binary)

                results[f"Fold {i+1}-H{h_idx+1}-L{l_idx+1}"] = {
                    "hidden": hidden,
                    "lam": l,
                    "acc": acc,
                    "f1": f1,
                    "model": model
                }

    # 이후 저장 및 시각화는 그대로

    # 📋 저장: 테이블 이미지로
    save_metrics_table({dataset_name: results}, "evaluation_extra3")

    # 🌟 가장 좋은 모델 하나에 대해 학습곡선 저장
    plot_best_learning_curve(results, dataset_name, "evaluation_extra3")

    # 📊 판다스 DataFrame으로 저장
    records = []
    for key, val in results.items():
        records.append({
            "Fold": key,
            "Hidden Layers": str(val['hidden']),
            "Lambda": val['lam'],
            "Accuracy": val['acc'],
            "F1 Score": val['f1']
        })
    print(f"✅ Saved DataFrame to evaluation/{dataset_name.lower()}_results.csv")

if __name__ == "__main__":
    main()