import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# ===== 활성화 함수 =====
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

# ===== 신경망 클래스 =====
class NeuralNetwork:
    def __init__(self, layer_sizes, alpha=0.01, lam=0.0):
        self.layer_sizes = layer_sizes
        self.alpha = alpha
        self.lam = lam
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            l_in = self.layer_sizes[i] + 1  # +1 for bias
            l_out = self.layer_sizes[i + 1]
            weight = np.random.randn(l_out, l_in) * np.sqrt(2 / l_in)
            weights.append(weight)
        return weights

    def forward_propagation(self, X):
        a = X
        activations = [a]
        zs = []

        for W in self.weights:
            a = np.insert(a, 0, 1, axis=1)  # Add bias unit
            z = a @ W.T
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        return activations, zs

    def compute_cost(self, y_pred, y_true, m):
        term = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        cost = np.sum(term) / m

        reg = 0
        for W in self.weights:
            reg += np.sum(W[:, 1:] ** 2)
        cost += (self.lam / (2 * m)) * reg
        return cost

    def backpropagation(self, X, y):
        m = X.shape[0]
        Delta = [np.zeros(W.shape) for W in self.weights]

        A, Z = self.forward_propagation(X)
        delta = A[-1] - y  # Output layer error

        for l in reversed(range(len(self.weights))):
            a_prev = np.insert(A[l], 0, 1, axis=1)
            Delta[l] += delta.T @ a_prev

            if l > 0:
                W_no_bias = self.weights[l][:, 1:]
                delta = (delta @ W_no_bias) * sigmoid_gradient(Z[l - 1])

        gradients = []
        for i in range(len(self.weights)):
            grad = Delta[i] / m
            grad[:, 1:] += (self.lam / m) * self.weights[i][:, 1:]
            gradients.append(grad)

        return gradients

    def update_weights(self, gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= self.alpha * gradients[i]

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            grads = self.backpropagation(X, y)
            self.update_weights(grads)
            if epoch % 10 == 0:
                y_pred = self.predict(X)
                cost = self.compute_cost(y_pred, y, X.shape[0])
                print(f"Epoch {epoch} - Cost: {cost:.4f}")

    def predict(self, X):
        A, _ = self.forward_propagation(X)
        return A[-1]

# ===== One-Hot Encoding 함수 =====
def one_hot_encode_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column].values.reshape(-1, 1)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_cols)
        ]
    )

    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y

# ===== 데이터셋 로딩 =====
def load_wdbc_dataset(path):
    df = pd.read_csv(path)

    # ✅ 'label' 컬럼이 정답 (Malignant/Benign)
    y = (df['label'] == 'M').astype(int).values.reshape(-1, 1)  # Malignant=1, Benign=0

    # ✅ 'id', 'label' 컬럼 제거하고 feature만 남기기
    X = df.drop(columns=['label']).values

    # ✅ 정규화 (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def load_loan_dataset(path):
    df = pd.read_csv(path)
    X_encoded, y = one_hot_encode_data(df, target_column='label')
    return X_encoded, y

# ===== Stratified K-Fold Cross Validation =====
def evaluate_model(X, y, hidden_layers, alpha=0.01, lam=0.0, epochs=100, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        input_size = X_train.shape[1]
        output_size = 1
        layer_sizes = [input_size] + hidden_layers + [output_size]

        model = NeuralNetwork(layer_sizes, alpha=alpha, lam=lam)
        model.fit(X_train, y_train, epochs=epochs)

        y_pred = model.predict(X_test)
        y_pred_label = (y_pred >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred_label)
        f1 = f1_score(y_test, y_pred_label)

        accuracies.append(acc)
        f1_scores.append(f1)

    return np.mean(accuracies), np.mean(f1_scores)

# ===== 실행 예시 =====
if __name__ == "__main__":
    # WDBC 데이터
    X_wdbc, y_wdbc = load_wdbc_dataset("wdbc.csv")

    # Loan 데이터
    X_loan, y_loan = load_loan_dataset("loan.csv")

    # 모델 설정
    hidden_layers = [10, 5]  # 두 hidden layer: 10 neurons -> 5 neurons
    alpha = 0.1
    lam = 0.01
    epochs = 100

    print("==== WDBC Dataset ====")
    acc_wdbc, f1_wdbc = evaluate_model(X_wdbc, y_wdbc, hidden_layers, alpha, lam, epochs)
    print(f"WDBC Accuracy: {acc_wdbc:.4f}, F1 Score: {f1_wdbc:.4f}")

    print("\n==== Loan Dataset ====")
    acc_loan, f1_loan = evaluate_model(X_loan, y_loan, hidden_layers, alpha, lam, epochs)
    print(f"Loan Accuracy: {acc_loan:.4f}, F1 Score: {f1_loan:.4f}")
