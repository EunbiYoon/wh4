import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
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
        self.accuracy = []
        self.f1 = []

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
        fold = pd.concat([c0, c1]).sample(frac=1).reset_index(drop=True)
        folds.append(fold)
    return folds

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
        col_labels = ["Layer & Neuron", "Lambda", "Accuracy", "F1 Score"]
        cell_data = []
        for key, val in dataset_results.items():
            cell_data.append([
                str(val['hidden']),
                f"{val['lam']:.2f}",
                f"{val['acc']:.4f}",
                f"{val['f1']:.4f}"
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
    dataset_name = DATASET_NAME
    alpha = 0.01
    lam = 0.25
    hidden_layers = [8, 6]  # 예시

    results = {}

    for fold_index in range(5):
        test_fold = folds[fold_index]
        train_folds = [f for i, f in enumerate(folds) if i != fold_index]
        train_df = pd.concat(train_folds)

        X_train = train_df.drop(columns=['label']).values
        y_train = train_df['label'].values.reshape(-1, 1)
        X_test = test_fold.drop(columns=['label']).values
        y_test = test_fold['label'].values.astype(int).ravel()  # ✅ 수정

        model = NeuralNetwork(layer_sizes=[X_train.shape[1], *hidden_layers, 1], alpha=alpha, lam=lam)
        model.fit(X_train, y_train, epochs=100, fold_index=fold_index)
        preds = model.predict(X_test)
        preds_binary = (preds >= 0.5).astype(int).ravel()  # ✅ 수정

        acc = accuracy_score(y_test, preds_binary)
        f1 = f1_score(y_test, preds_binary)
        print(f"✅ Fold {fold_index+1} - Accuracy: {acc:.4f}, F1: {f1:.4f}")

        results[f"Fold {fold_index+1}"] = {
            "hidden": hidden_layers,
            "lam": lam,
            "acc": acc,
            "f1": f1
        }

        plot_learning_curve(model, dataset_name, hidden_layers, lam, fold_index)

    save_metrics_table({dataset_name: results})

# === Entry Point ===
if __name__ == "__main__":
    main()
