import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from propagation import backpropagation, forward_propagation, cost_function

# === Dataset Setting ===
DATASET_NAME = "wdbc"

# === Neural Network Class ===
class NeuralNetwork:
    def __init__(self, layer_sizes, alpha=0.01, lam=0.0):
        self.layer_sizes = layer_sizes
        self.alpha = alpha
        self.lam = lam
        self.weights = self.initialize_weights()
        self.cost_history = []

    def initialize_weights(self):
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            l_in = self.layer_sizes[i] + 1
            l_out = self.layer_sizes[i + 1]
            weight = np.random.randn(l_out, l_in) * np.sqrt(2 / l_in)
            weights.append(weight)
        return weights

    def update_weights(self, gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * gradients[i]

    def fit(self, X, y, epochs=100, fold_index=None):
        for epoch in range(epochs):
            all_a_lists, _ = forward_propagation(self.weights, X)
            finalized_D, _, _ = backpropagation(self.weights, all_a_lists, y, self.lam)
            self.update_weights(finalized_D)
            pred_ys = [a_list[-1] for a_list in all_a_lists]
            _, final_cost = cost_function(pred_ys, y, self.weights, self.lam)
            self.cost_history.append(final_cost)
            if epoch % 10 == 0:
                prefix = f"[Fold {fold_index}] " if fold_index is not None else ""
                print(f"{prefix}Epoch {epoch} - Cost: {final_cost:.4f}")

    def predict(self, X):
        all_a_lists, _ = forward_propagation(self.weights, X)
        return np.array([a_list[-1] for a_list in all_a_lists])

# === Data Loading ===
def load_dataset():
    DATA_PATH = f"datasets/{DATASET_NAME}.csv"
    df = pd.read_csv(DATA_PATH)
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")
    y = df['label'].copy()
    X = df.drop(columns=['label'])

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

    if y.dtype == 'O':
        y = (y == 'M').astype(int)
    else:
        y = pd.to_numeric(y)
    return X.values, y.values.reshape(-1, 1)

# === Stratified K-Fold ===
def stratified_k_fold_split(X, y, k=5):
    df = pd.DataFrame(X)
    df['label'] = y.ravel()

    class_0 = df[df['label'] == 0].sample(frac=1).reset_index(drop=True)
    class_1 = df[df['label'] == 1].sample(frac=1).reset_index(drop=True)

    folds = []
    for i in range(k):
        c0 = class_0.iloc[int(len(class_0)*i/k):int(len(class_0)*(i+1)/k)]
        c1 = class_1.iloc[int(len(class_1)*i/k):int(len(class_1)*(i+1)/k)]
        test_df = pd.concat([c0, c1]).sample(frac=1).reset_index(drop=True)

        remaining_c0 = pd.concat([class_0.iloc[:int(len(class_0)*i/k)], class_0.iloc[int(len(class_0)*(i+1)/k):]])
        remaining_c1 = pd.concat([class_1.iloc[:int(len(class_1)*i/k)], class_1.iloc[int(len(class_1)*(i+1)/k):]])
        train_df = pd.concat([remaining_c0, remaining_c1]).sample(frac=1).reset_index(drop=True)

        folds.append((train_df, test_df))
    return folds
def my_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def my_f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)

# === Plot Learning Curve ===
def plot_learning_curve(model, dataset_name, hidden_layer, lam, fold_index=None):
    os.makedirs("evaluation", exist_ok=True)
    plt.figure()
    plt.plot(model.cost_history, marker='o')
    total_neurons = sum(hidden_layer)
    title = f"{dataset_name} Learning Curve\nλ={lam}, Layers={len(hidden_layer)}, Neurons={total_neurons}"
    if fold_index is not None:
        title += f", Fold {fold_index+1}"
    plt.title(title, fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.tight_layout()
    filename = f"evaluation/{dataset_name.lower()}_curve_fold{fold_index+1}.png"
    plt.savefig(filename)
    print(f"📉 Saved: {filename}")
    plt.close()

# === Save Metrics Table ===
def save_metrics_table(results_by_dataset):
    os.makedirs("evaluation", exist_ok=True)
    for dataset_name, dataset_results in results_by_dataset.items():
        fig, ax = plt.subplots()
        ax.axis('off')
        col_labels = ["Layer & Neuron", "Lambda", "Avg Accuracy", "Avg F1 Score"]
        cell_data = []

        # 새로운 구조: key = (hidden, lam)
        grouped = {}
        for key, val in dataset_results.items():
            h = tuple(val['hidden'])  # list는 dict 키로 못 씀
            l = val['lam']
            grouped.setdefault((h, l), []).append((val['acc'], val['f1']))

        # 평균 계산 후 테이블 행 추가
        for (h, l), metrics in grouped.items():
            accs = [m[0] for m in metrics]
            f1s = [m[1] for m in metrics]
            avg_acc = np.mean(accs)
            avg_f1 = np.mean(f1s)
            cell_data.append([
                str(h),
                f"{l:.2f}",
                f"{avg_acc:.4f}",
                f"{avg_f1:.4f}"
            ])

        table = ax.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.1, 1.6)
        plt.title(f"{dataset_name} Model Performance", fontweight='bold')
        plt.tight_layout()
        filename = f"evaluation/{dataset_name.lower()}_table.png"
        plt.savefig(filename)
        print(f"📋 Saved metrics table: {filename}")
        plt.close()


# === Main Function ===
def main():
    X, y = load_dataset()
    folds = stratified_k_fold_split(X, y, k=5)

    lam = [0.25, 10]
    hidden_layers = [[8, 6], [10,10,10]]
    alpha = 0.01
    epochs = 100
    dataset_name = DATASET_NAME

    results = {}

    for h_idx, hidden in enumerate(hidden_layers):
        for l_idx, l in enumerate(lam):
            for i, (train_df, test_df) in enumerate(folds):
                X_train = train_df.drop(columns=['label']).values
                y_train = train_df['label'].values.reshape(-1, 1)
                X_test = test_df.drop(columns=['label']).values
                y_test = test_df['label'].values.astype(int).ravel()

                model = NeuralNetwork(layer_sizes=[X_train.shape[1], *hidden, 1], alpha=alpha, lam=l)
                model.fit(X_train, y_train, epochs=epochs, fold_index=i)
                preds = model.predict(X_test)
                preds_binary = (preds >= 0.5).astype(int).ravel()

                acc = my_accuracy(y_test, preds_binary)
                f1 = my_f1_score(y_test, preds_binary)

                results[f"Fold {i+1}-H{h_idx+1}-L{l_idx+1}"] = {
                    "hidden": hidden,
                    "lam": l,
                    "acc": acc,
                    "f1": f1
                }

                plot_learning_curve(model, dataset_name, hidden, l, fold_index=i)


    save_metrics_table({dataset_name: results})

if __name__ == "__main__":
    main()
