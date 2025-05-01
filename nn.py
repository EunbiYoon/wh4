# === Vectorized Neural Network Implementation ===
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from propagation import backpropagation_vectorized, forward_propagation, cost_function
import debug_text

# === Setting ===
DATASET_NAME = "titanic"  # Name of the dataset
M_SIZE = 300            # m for stopping criterion
DEBUG_MODE = True         # If True, run debugging routine at the end
TRAIN_MODE = "mini-batch"    # Choose "batch" or "mini-batch"
BATCH_SIZE = 64

# === FILE_NAME Setting ===
if TRAIN_MODE=="batch":
    FILE_NAME = DATASET_NAME
elif TRAIN_MODE=="mini-batch":
    FILE_NAME = DATASET_NAME+"_minibatch"
else:
    print("choose mini-batch or batch in TRAIN_MODE")

# === Neural Network Class ===
class NeuralNetwork:
    def __init__(self, layer_sizes, alpha=0.01, lambda_reg=0.0):
        self.layer_sizes = layer_sizes  # Architecture of the network
        self.alpha = alpha              # Learning rate
        self.lambda_reg = lambda_reg    # Regularization parameter
        self.weights = self.initialize_weights()  # Random weight initialization
        self.cost_history = []          # Store J value per training set

    def initialize_weights(self):
        # Initialize weights for each layer using uniform distribution
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            l_in = self.layer_sizes[i] + 1  # +1 to account for bias unit
            l_out = self.layer_sizes[i + 1]  # Output layer size
            weight = np.random.uniform(-1, 1, size=(l_out, l_in))  # Random initialization : random numbers from -1 to +1
            weights.append(weight)
        return weights

    def update_weights(self, gradients):
        # Update weights using gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * gradients[i]

    def fit(self, X, y, batch_size=32, fold_index=None, mode='batch', stopping_J=600):
        m = X.shape[0]  # Total number of samples
        m_size = 0      # Epoch counter

        while True:
            if mode == 'mini-batch':
                # Shuffle data
                indices = np.arange(m)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                # Process mini-batches
                for start in range(0, m, batch_size):
                    end = start + batch_size
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    # Forward and backward pass
                    A, Z, _, _ = forward_propagation(self.weights, X_batch)
                    finalized_D = backpropagation_vectorized(self.weights, A, Z, y_batch, self.lambda_reg)
                    self.update_weights(finalized_D)

            elif mode == 'batch':
                # Shuffle data
                indices = np.arange(m)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                # Full batch forward/backward pass
                A, Z, _, _ = forward_propagation(self.weights, X_shuffled)
                finalized_D = backpropagation_vectorized(self.weights, A, Z, y_shuffled, self.lambda_reg)
                self.update_weights(finalized_D)

            else:
                raise ValueError("Mode must be either 'batch' or 'mini-batch'")

            # Compute cost on full dataset
            A, _, _, _ = forward_propagation(self.weights, X)
            _, final_cost = cost_function(A[-1], y, self.weights, self.lambda_reg)
            self.cost_history.append(final_cost)

            # Print intermediate results
            prefix = f"[Fold {fold_index}] " if fold_index is not None else ""
            model_info = f"Hidden={self.layer_sizes[1:-1]}, λ={self.lambda_reg}, dataset={DATASET_NAME}"
            if m_size % 10 == 0:
                print(f"{prefix}Epoch {m_size} - Cost: {final_cost:.8f} - {model_info}")

            # Stop training if max m_size reached
            if m_size == stopping_J:
                print(f"{prefix}Stopping at m_size {m_size} - Final Cost J: {final_cost:.8f}")
                # save for debug
                self.last_A = A
                self.last_Z = Z
                self.finalized_D = finalized_D
                self.final_cost = final_cost
                break

            m_size += 1

    def predict(self, X):
        # Return output layer activation as prediction
        A, _, _, _ = forward_propagation(self.weights, X)
        return A[-1]

# === Load dataset and apply preprocessing ===
def load_dataset():
    # Load dataset from CSV file
    DATA_PATH = f"datasets/{DATASET_NAME}.csv"
    df = pd.read_csv(DATA_PATH)
    if 'label' not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    y = df['label'].copy()
    X = df.drop(columns=['label'])

    # Normalize numeric columns and one-hot encode categorical columns
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
    # Create stratified folds with equal class distribution
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
    # Identify model with lowest final cost
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

    # X-axis: iteration * number of training instances
    x_vals = [i * train_size for i in range(len(model.cost_history))]
    y_vals = model.cost_history

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

    filename = f"{save_folder}/{FILE_NAME.lower()}_best_curve.png"
    plt.savefig(filename)
    print(f"🌟 Saved best learning curve: {filename}")
    plt.close()

# === Save Metrics Table as Image ===
def save_metrics_table(results_by_dataset, save_folder):
    os.makedirs("evaluation", exist_ok=True)
    for dataset_name, dataset_results in results_by_dataset.items():
        fig, ax = plt.subplots()
        ax.axis('off')

        # Determine table columns based on whether mini-batch was used
        if any(val['mode'] == 'mini-batch' for val in dataset_results.values()):
            col_labels = ["Layer & Neuron", "Lambda", "Alpha", "Batch Size", "Mode", "Avg Accuracy", "Avg F1 Score"]
            show_batch_size = True
        else:
            col_labels = ["Layer & Neuron", "Lambda", "Alpha", "Mode", "Avg Accuracy", "Avg F1 Score"]
            show_batch_size = False

        cell_data = []
        grouped = {}

        # Group by configuration for averaging
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

            row = [str(h), f"{l}", f"{a:.3f}"]
            if show_batch_size:
                row.append(str(b))
            row.extend([m, f"{avg_acc:.4f}", f"{avg_f1:.4f}"])
            cell_data.append(row)

        table = ax.table(cellText=cell_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.1, 1.6)

        plt.title(f"{dataset_name} Model Performance", fontweight='bold')
        plt.tight_layout()
        filename = f"{save_folder}/{FILE_NAME.lower()}_table.png"
        plt.savefig(filename)
        print(f"📋 Saved metrics table: {filename}")
        plt.close()


# === Neural Network Execution Wrapper ===
def neural_network():
    X, y = load_dataset()  # Load data and labels
    folds = stratified_k_fold_split(X, y, k=5)  # Create 5-fold split

    lambda_reg_list = [0.1, 0.000001]  # List of λ values to test
    hidden_layers = [[32], [32, 16], [32, 16, 8], [32, 16, 8, 4]]  # Layer architectures to test
    alpha = 0.1            # Learning rate
    batch_size = BATCH_SIZE        # Size of mini-batches
    mode = TRAIN_MODE        # Training mode

    dataset_name = DATASET_NAME
    results = {}  # Dictionary to collect evaluation metrics

    for h_idx, hidden in enumerate(hidden_layers):
        for l_idx, lambda_reg in enumerate(lambda_reg_list):
            for i, (train_df, test_df) in enumerate(folds):
                X_train = train_df.drop(columns=['label']).values
                y_train = train_df['label'].values.reshape(-1, 1)
                X_test = test_df.drop(columns=['label']).values
                y_test = test_df['label'].values.astype(int).ravel()

                # Initialize model with specific structure and parameters
                model = NeuralNetwork(
                    layer_sizes=[X_train.shape[1], *hidden, 1],
                    alpha=alpha,
                    lambda_reg=lambda_reg
                )

                # Train the model
                model.fit(
                    X_train, y_train,
                    batch_size=batch_size,
                    fold_index=i,
                    mode=mode,
                    stopping_J=M_SIZE
                )

                # Make predictions
                preds = model.predict(X_test)
                preds_binary = (preds >= 0.5).astype(int).ravel()

                # Evaluate performance
                acc = my_accuracy(y_test, preds_binary)
                f1 = my_f1_score(y_test, preds_binary)

                # Store results
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

    # Save performance table and learning curve
    save_metrics_table({dataset_name: results}, "evaluation")
    plot_best_learning_curve(results, dataset_name, "evaluation")

    # Run debugging output if flag is enabled
    if DEBUG_MODE == True:
        A, Z, _, _ = forward_propagation(model.weights, X_train)
        finalized_D = model.finalized_D
        final_cost = model.final_cost

        # Now safely extract all a_lists and z_lists for every instance
        all_a_lists = [[a[i].reshape(-1, 1) for a in A] for i in range(X_train.shape[0])]
        all_z_lists = [[z[i].reshape(-1, 1) for z in Z] for i in range(X_train.shape[0])]
        pred_y_list = [a_list[-1] for a_list in all_a_lists]
        true_y_list = [y_train[i].reshape(-1, 1) for i in range(y_train.shape[0])]
        J_list = [-(yt.T @ np.log(yp) + (1 - yt).T @ np.log(1 - yp)).item()
                for yt, yp in zip(true_y_list, pred_y_list)]

        # delta_list, D_list calculation
        delta_list = []
        D_list = []
        for i in range(X_train.shape[0]):
            deltas = [None] * len(model.weights)
            deltas[-1] = all_a_lists[i][-1] - y_train[i].reshape(-1, 1)
            for l in reversed(range(len(model.weights) - 1)):
                a = all_a_lists[i][l + 1][1:]  # exclude bias
                da = a * (1 - a)
                deltas[l] = (model.weights[l + 1][:, 1:].T @ deltas[l + 1]) * da
            delta_list.append(deltas)
            D_list.append([deltas[l] @ all_a_lists[i][l].T for l in range(len(model.weights))])

        # call debug function
        debug_text.main(lambda_reg, X_train, y_train, model.weights, all_a_lists, all_z_lists,
                J_list, final_cost, delta_list, D_list, finalized_D, FILE_NAME)
        
# === Main Entry Point ===
if __name__ == "__main__":
    neural_network()