import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8))

    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(cost)
    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(w, b, X, Y, num_iters, alpha, print_cost=False):
    costs = []
    for i in range(num_iters):
        grads, cost = propagate(w, b, X, Y)
        dw, db = grads["dw"], grads["db"]

        w -= alpha * dw
        b -= alpha * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost:.6f}")

    return {"w": w, "b": b}, {"dw": dw, "db": db}, costs


def predict(w, b, X):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    y_pred = (A > 0.5).astype(int)
    return y_pred


def evaluate_metrics(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    tpr = TP / (TP + FN + 1e-8) * 100
    fpr = FP / (FP + TN + 1e-8) * 100
    precision = TP / (TP + FP + 1e-8) * 100
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8) * 100

    print(f"True Positive:  {TP}")
    print(f"True Negative:  {TN}")
    print(f"False Negative: {FN}")
    print(f"False Positive: {FP}")
    print(f"True Positive Rate / Recall: {tpr:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"False Positive Rate / Fallout: {fpr:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%\n")


def model(X_train, Y_train, num_iters=2000, alpha=0.5, print_cost=False):
    w, b = initialize(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iters, alpha, print_cost)

    w, b = parameters["w"], parameters["b"]

    # Load and preprocess test data
    X_test = pd.read_csv("test_cancer_data.csv").to_numpy().T
    Y_test = pd.read_csv("test_cancer_data_y.csv").to_numpy().reshape(1, -1)

    # Predict
    y_prediction_train = predict(w, b, X_train)
    y_prediction_test = predict(w, b, X_test)

    print("Performance on Training Set:")
    evaluate_metrics(Y_train, y_prediction_train)

    print("Performance on Test Set:")
    evaluate_metrics(Y_test, y_prediction_test)

    # Plot cost curve
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per 100s)')
    plt.title(f"Learning rate = {alpha}")
    plt.grid(True)
    plt.show()

    return {
        "costs": costs,
        "Y_prediction_test": y_prediction_test,
        "Y_prediction_train": y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": alpha,
        "num_iterations": num_iters
    }


def main():
    # Load and preprocess training data
    train_x = pd.read_csv("cancer_data.csv").to_numpy().T
    train_y = pd.read_csv("cancer_data_y.csv").to_numpy().reshape(1, -1)

    # Train the model
    d = model(train_x, train_y, num_iters=190500, alpha=6.5e-8, print_cost=True)


if __name__ == "__main__":
    main()
