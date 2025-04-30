import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from propagation import backpropagation, forward_propagation, cost_function

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

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            all_a_lists, _ = forward_propagation(self.weights, X)
            finalized_D, _, _ = backpropagation(self.weights, all_a_lists, y, self.lam)
            self.update_weights(finalized_D)
            pred_ys = [a_list[-1] for a_list in all_a_lists]
            _, final_cost = cost_function(pred_ys, y, self.weights, self.lam)
            self.cost_history.append(final_cost)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Cost: {final_cost:.4f}")

    def predict(self, X):
        all_a_lists, _ = forward_propagation(self.weights, X)
        return np.array([a_list[-1] for a_list in all_a_lists])

# === Data Loading ===
def one_hot_encode_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column].values.reshape(-1, 1)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ])
    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y

def load_wdbc_dataset(path):
    df = pd.read_csv(path)
    y = (df['label'] == 'M').astype(int).values.reshape(-1, 1)
    X = df.drop(columns=['label']).values
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y

def load_loan_dataset(path):
    df = pd.read_csv(path)
    return one_hot_encode_data(df, target_column='label')

# === Plot Learning Curve ===
def plot_learning_curve(model, dataset_name, hidden_layer, lam):
    plt.figure()
    plt.plot(model.cost_history, marker='o')
    num_layers = len(hidden_layer)
    total_neurons = sum(hidden_layer)
    plt.title(f"{dataset_name} Learning Curve\n"
              f"λ={lam}, Layers={num_layers}, Neurons={total_neurons}", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.tight_layout()
    filename = f"learning_curve/{dataset_name.lower()}_curve.png"
    plt.savefig(filename)
    print(f"📉 Saved: {filename}")

# === Save Accuracy/F1 Table ===
def save_metrics_table_as_image(results_by_dataset):
    for dataset_name, dataset_results in results_by_dataset.items():
        fig, ax = plt.subplots()
        ax.axis('off')
        col_labels = ["Layer & Neuron", "Lambda", "Accuracy", "F1 Score"]
        cell_data = []
        for key, val in dataset_results.items():
            layer_idx = int(key.split('_')[0][1:]) - 1
            hidden_layer = hidden_layers_list[layer_idx // len(lambda_reg_list)]
            lam = lambda_reg_list[layer_idx % len(lambda_reg_list)]
            cell_data.append([
                str(hidden_layer),
                f"{lam:.2f}",
                f"{val['acc']:.4f}",
                f"{val['f1']:.4f}"
            ])
        table = ax.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.1, 1.6)
        plt.title(f"{dataset_name} Model Performance", fontweight='bold')
        plt.tight_layout()
        filename = f"learning_curve/{dataset_name.lower()}_table.png"
        plt.savefig(filename)
        print(f"📋 Saved metrics table: {filename}")

# === Main Execution ===
if __name__ == "__main__":
    os.makedirs("learning_curve", exist_ok=True)
    hidden_layers_list = [[2], [4, 4, 4], [6, 6, 6, 6]]
    lambda_reg_list = [0.01, 0.1]
    alpha = 0.1
    epochs = 100

    for dataset_name, (X, y) in {"WDBC": load_wdbc_dataset("wdbc.csv"), "Loan": load_loan_dataset("loan.csv")}.items():
        results_by_dataset = {dataset_name: {}}
        results_with_model = {}
        config_idx = 0

        for i, hidden_layer in enumerate(hidden_layers_list):
            for j, lam in enumerate(lambda_reg_list):
                print(f"\n📘 {dataset_name}: hidden={hidden_layer}, λ={lam}")
                model = NeuralNetwork([X.shape[1]] + hidden_layer + [1], alpha=alpha, lam=lam)
                model.fit(X, y, epochs=epochs)

                y_pred = (model.predict(X) >= 0.5).astype(int).ravel()
                y_true = y.ravel()
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=1)

                config_name = f"h{config_idx+1}_λ{lam}"
                results_by_dataset[dataset_name][config_name] = {"acc": acc, "f1": f1}
                results_with_model[f"{dataset_name}_{config_name}"] = {
                    "model": model,
                    "dataset_name": dataset_name,
                    "acc": acc,
                    "f1": f1
                }
                config_idx += 1

        # === Best model (per dataset)
        filtered = {k: v for k, v in results_with_model.items() if v['dataset_name'] == dataset_name}
        best_key = max(filtered.items(), key=lambda x: (x[1]['f1'], x[1]['acc']))[0]
        best_model = results_with_model[best_key]["model"]
        layer_idx = int(best_key.split('_')[1][1:]) - 1
        best_hidden_layer = hidden_layers_list[layer_idx // len(lambda_reg_list)]
        best_lam = lambda_reg_list[layer_idx % len(lambda_reg_list)]
        print(f"\n🎯 Best {dataset_name} Model: {best_key}")
        plot_learning_curve(best_model, dataset_name.lower() + "_best", best_hidden_layer, best_lam)

        # === Save performance table (per dataset)
        save_metrics_table_as_image(results_by_dataset)
