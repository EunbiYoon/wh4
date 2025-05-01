# === Vectorized Neural Network Implementation ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from propagation import backpropagation_vectorized, forward_propagation, cost_function, run_debug

# === Setting ===
DATASET_NAME = "raisin"
M_SIZE = 10
DEBUG_MODE = True

# === Neural Network Class ===
class NeuralNetwork:
    def __init__(self, layer_sizes, alpha=0.01, lambda_reg=0.0):
        self.layer_sizes = layer_sizes
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.weights = self.initialize_weights()
        self.cost_history = []

    def initialize_weights(self):
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            l_in = self.layer_sizes[i] + 1
            l_out = self.layer_sizes[i + 1]
            weight = np.random.uniform(-1, 1, size=(l_out, l_in))
            weights.append(weight)
        return weights

    def update_weights(self, gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * gradients[i]

    def fit(self, X, y, batch_size=32, fold_index=None, mode='batch', stopping_J=600):
        m = X.shape[0]
        m_size = 0

        while True:
            if mode == 'mini-batch':
                indices = np.arange(m)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                for start in range(0, m, batch_size):
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    A, Z, _, _ = forward_propagation(self.weights, X_batch)
                    finalized_D = backpropagation_vectorized(self.weights, A, Z, y_batch, self.lambda_reg)
                    self.update_weights(finalized_D)

            elif mode == 'batch':
                indices = np.arange(m)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                A, Z, _, _ = forward_propagation(self.weights, X_shuffled)
                finalized_D = backpropagation_vectorized(self.weights, A, Z, y_shuffled, self.lambda_reg)
                self.update_weights(finalized_D)
            else:
                raise ValueError("Mode must be either 'batch' or 'mini-batch'")

            A, _, _, _ = forward_propagation(self.weights, X)
            _, final_cost = cost_function(A[-1], y, self.weights, self.lambda_reg)
            self.cost_history.append(final_cost)

            prefix = f"[Fold {fold_index}] " if fold_index is not None else ""
            model_info = f"Hidden={self.layer_sizes[1:-1]}, λ={self.lambda_reg}, dataset={DATASET_NAME}"
            if m_size % 10 == 0:
                print(f"{prefix}Epoch {m_size} - Cost: {final_cost:.8f} - {model_info}")

            if m_size == stopping_J:
                print(f"{prefix}Stopping at m_size {m_size} - Final Cost J: {final_cost:.8f}")
                break

            m_size += 1

    def predict(self, X):
        A, _, _, _ = forward_propagation(self.weights, X)
        return A[-1]

# === Load dataset and apply preprocessing ===
def load_dataset():
    DATA_PATH = f"datasets/{DATASET_NAME}.csv"
    df = pd.read_csv(DATA_PATH)
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    y = df['label'].copy()
    X = df.drop(columns=['label'])

    # Normalize numeric columns and apply one-hot encoding to categorical columns
    for col in X.columns:
        if col.endswith("_num"):
            mean = X[col].mean()
            std = X[col].std()
            X[col] = (X[col] - mean) / std
        elif col.endswith("_cat"):
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(X[[col]])
            encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{i}" for i in range(encoded.shape[1])])
            X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)

    return X.values, y.values.reshape(-1, 1)

# === Stratified K-Fold Split ===
def stratified_k_fold_split(X, y, k=5):
    df = pd.DataFrame(X)
    df['label'] = y.ravel()
    class_0 = df[df['label'] == 0].sample(frac=1).reset_index(drop=True)
    class_1 = df[df['label'] == 1].sample(frac=1).reset_index(drop=True)
    folds = []
    for i in range(k):
        # test data
        c0 = class_0.iloc[int(len(class_0)*i/k):int(len(class_0)*(i+1)/k)]
        c1 = class_1.iloc[int(len(class_1)*i/k):int(len(class_1)*(i+1)/k)]
        test_df = pd.concat([c0, c1]).sample(frac=1).reset_index(drop=True)
        # train data
        remaining_c0 = pd.concat([class_0.iloc[:int(len(class_0)*i/k)], class_0.iloc[int(len(class_0)*(i+1)/k):]])
        remaining_c1 = pd.concat([class_1.iloc[:int(len(class_1)*i/k)], class_1.iloc[int(len(class_1)*(i+1)/k):]])
        train_df = pd.concat([remaining_c0, remaining_c1]).sample(frac=1).reset_index(drop=True)
        folds.append((train_df, test_df))
    return folds

# === Accuracy Calculation ===
def my_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

# === F1 Score Calculation ===
def my_f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# === Plot Best Model's Learning Curve ===
def plot_best_learning_curve(results, dataset_name, save_folder):
    best_key = min(results, key=lambda k: results[k]['model'].cost_history[-1])
    best_info = results[best_key]
    model = best_info['model']
    hidden_layer = best_info['hidden']
    lambda_reg = best_info['lambda_reg']
    train_size = best_info['train_size']
    alpha = best_info['alpha']
    mode = best_info['mode']
    batch_size = best_info['batch_size']

    os.makedirs("evaluation", exist_ok=True)

    # X축: m_size * train_size
    x_vals = [i * train_size for i in range(len(model.cost_history))]
    y_vals = model.cost_history

    # 조건부 설명 텍스트 만들기
    info_text = f"λ={lambda_reg},  Hidden={hidden_layer}, α={alpha}, Mode={mode}"
    if mode == "mini-batch":
        info_text += f", Batch Size={batch_size}"

    plt.figure()
    plt.plot(x_vals, y_vals, marker='o')
    plt.title(f"{dataset_name.capitalize()} BEST Learning Curve\n{info_text}", fontsize=11)
    plt.xlabel("Training Instances (m x Train Set)")
    plt.ylabel("Cost (J)")
    plt.grid(True)
    plt.tight_layout()

    filename = f"{save_folder}/{dataset_name.lower()}_best_curve.png"
    plt.savefig(filename)
    print(f"🌟 Saved best learning curve: {filename}")
    plt.close()



# === Save Metrics Table as Image ===
def save_metrics_table(results_by_dataset, save_folder):
    os.makedirs("evaluation", exist_ok=True)
    for dataset_name, dataset_results in results_by_dataset.items():
        fig, ax = plt.subplots()
        ax.axis('off')

        # ✅ 조건부로 Batch Size 컬럼 추가
        if any(val['mode'] == 'mini-batch' for val in dataset_results.values()):
            col_labels = ["Layer & Neuron", "Lambda", "Alpha", "Batch Size", "Mode", "Avg Accuracy", "Avg F1 Score"]
            show_batch_size = True
        else:
            col_labels = ["Layer & Neuron", "Lambda", "Alpha", "Mode", "Avg Accuracy", "Avg F1 Score"]
            show_batch_size = False

        cell_data = []
        grouped = {}

        for key, val in dataset_results.items():
            h = tuple(val['hidden'])
            l = val['lambda_reg']
            a = val['alpha']
            b = val['batch_size']
            m = val['mode']
            grouped.setdefault((h, l, a, b, m), []).append((val['acc'], val['f1']))

        for (h, l, a, b, m), metrics in grouped.items():
            accs = [m[0] for m in metrics]
            f1s = [m[1] for m in metrics]
            avg_acc = np.mean(accs)
            avg_f1 = np.mean(f1s)

            row = [
                str(h),
                f"{l:.3f}",
                f"{a:.3f}",
            ]
            if show_batch_size:
                row.append(str(b))
            row.extend([
                m,
                f"{avg_acc:.4f}",
                f"{avg_f1:.4f}"
            ])
            cell_data.append(row)

        table = ax.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.1, 1.6)

        plt.title(f"{dataset_name} Model Performance", fontweight='bold')
        plt.tight_layout()
        filename = f"{save_folder}/{dataset_name.lower()}_table.png"
        plt.savefig(filename)
        print(f"📋 Saved metrics table: {filename}")
        plt.close()


# === Create Debug File Function ===
def neural_network():
    X, y = load_dataset()
    folds = stratified_k_fold_split(X, y, k=5)

    lambda_reg_list = [0.001, 0.0001, 0.00001, 0.000001]  # Regularization values
    hidden_layers = [[32],[32,16],[32,16,8]]  # Different architectures
    alpha = 0.1
    batch_size = 32
    mode = "batch"

    dataset_name = DATASET_NAME
    results = {}

    for h_idx, hidden in enumerate(hidden_layers):
        for l_idx, lambda_reg in enumerate(lambda_reg_list):
            for i, (train_df, test_df) in enumerate(folds):
                X_train = train_df.drop(columns=['label']).values
                y_train = train_df['label'].values.reshape(-1, 1)
                X_test = test_df.drop(columns=['label']).values
                y_test = test_df['label'].values.astype(int).ravel()

                model = NeuralNetwork(
                    layer_sizes=[X_train.shape[1], *hidden, 1],
                    alpha=alpha,
                    lambda_reg=lambda_reg
                )

                model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    fold_index=i,
                    mode=mode,
                    stopping_J=M_SIZE  # ← 추가된 인자
                )

                preds = model.predict(X_test)
                preds_binary = (preds >= 0.5).astype(int).ravel()
                acc = my_accuracy(y_test, preds_binary)
                f1 = my_f1_score(y_test, preds_binary)

                results[f"Fold {i+1}-H{h_idx+1}-L{l_idx+1}"] = {
                    "hidden": hidden,
                    "lambda_reg": lambda_reg,
                    "acc": acc,
                    "f1": f1,
                    "model": model,
                    "train_size": X_train.shape[0],
                    "alpha": alpha,
                    "batch_size": batch_size,
                    "mode": mode
                }

    save_metrics_table({dataset_name: results}, "evaluation")
    plot_best_learning_curve(results, dataset_name, "evaluation")
    records = []
    for key, val in results.items():
        records.append({
            "Fold": key,
            "Hidden Layers": str(val['hidden']),
            "Lambda": val['lambda_reg'],
            "Accuracy": val['acc'],
            "F1 Score": val['f1']
        })
    print(f"✅ Saved DataFrame to evaluation/{dataset_name.lower()}_results.csv")

    if DEBUG_MODE==True:    
        run_debug(model.weights, X_train, y_train, lambda_reg)


if __name__ == "__main__":
    neural_network()