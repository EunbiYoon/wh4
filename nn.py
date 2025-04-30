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
    def __init__(self, layer_sizes, alpha=0.01, lambda_reg=0.0):
        self.layer_sizes = layer_sizes  # Number of neurons in each layer
        self.alpha = alpha              # Learning rate
        self.lambda_reg = lambda_reg                  # Regularization parameter
        self.weights = self.initialize_weights()  # Initialize weights
        self.cost_history = []          # Store cost at each epoch

    # Initialize weights using He initialization
    def initialize_weights(self):
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            l_in = self.layer_sizes[i] + 1  # +1 for bias
            l_out = self.layer_sizes[i + 1]
            weight = np.random.uniform(-1, 1, size=(l_out, l_in))
            weights.append(weight)
        return weights

    # Update weights using gradients
    def update_weights(self, gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * gradients[i]

    # Fit the model using either batch or mini-batch gradient descent
    def fit(self, X, y, epochs=100, batch_size=32, fold_index=None, mode='batch'):
        m = X.shape[0]
        for epoch in range(epochs):
            if mode == 'mini-batch':
                indices = np.arange(m)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                for start in range(0, m, batch_size):
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]
                    all_a_lists, _ = forward_propagation(self.weights, X_batch)
                    finalized_D, _, _ = backpropagation(self.weights, all_a_lists, y_batch, self.lambda_reg)
                    self.update_weights(finalized_D)

            elif mode == 'batch':
                indices = np.arange(m)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                all_a_lists, _ = forward_propagation(self.weights, X)
                finalized_D, _, _ = backpropagation(self.weights, all_a_lists, y, self.lambda_reg)
                self.update_weights(finalized_D)
            else:
                raise ValueError("Mode must be either 'batch' or 'mini-batch')")

            all_a_lists, _ = forward_propagation(self.weights, X)
            pred_ys = [a_list[-1] for a_list in all_a_lists]
            _, final_cost = cost_function(pred_ys, y, self.weights, self.lambda_reg)
            self.cost_history.append(final_cost)
            if epoch % 10 == 0:
                prefix = f"[Fold {fold_index}] " if fold_index is not None else ""
                print(f"{prefix}Epoch {epoch} - Cost: {final_cost:.4f}")

    # Predict output for given input
    def predict(self, X):
        all_a_lists, _ = forward_propagation(self.weights, X)
        return np.array([a_list[-1] for a_list in all_a_lists])

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
        c0 = class_0.iloc[int(len(class_0)*i/k):int(len(class_0)*(i+1)/k)]
        c1 = class_1.iloc[int(len(class_1)*i/k):int(len(class_1)*(i+1)/k)]
        test_df = pd.concat([c0, c1]).sample(frac=1).reset_index(drop=True)
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

    os.makedirs("evaluation", exist_ok=True)
    plt.figure()
    plt.plot(model.cost_history, marker='o')
    title = f"{dataset_name} BEST Learning Curve\nλ={lambda_reg}, Hidden={hidden_layer}"
    plt.title(title, fontsize=12)
    plt.xlabel("Epoch")
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
        col_labels = ["Layer & Neuron", "Lambda", "Avg Accuracy", "Avg F1 Score"]
        cell_data = []
        grouped = {}
        for key, val in dataset_results.items():
            h = tuple(val['hidden'])
            l = val['lambda_reg']
            grouped.setdefault((h, l), []).append((val['acc'], val['f1']))
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
        filename = f"{save_folder}/{dataset_name.lower()}_table.png"
        plt.savefig(filename)
        print(f"📋 Saved metrics table: {filename}")
        plt.close()

# === Main Execution Function ===
def main():
    X, y = load_dataset()
    folds = stratified_k_fold_split(X, y, k=5)

    lambda_reg_list = [0.25, 0.5]  # Regularization values
    hidden_layers = [[8, 6], [8, 6, 4], [4]]  # Different architectures
    alpha = 0.01
    epochs = 100
    batch_size = 16
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
                    "lambda_reg": lambda_reg,
                    "acc": acc,
                    "f1": f1,
                    "model": model
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

if __name__ == "__main__":
    main()