import argparse
import os
import joblib
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd


def train_model(test_size, random_state):
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # Ensure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    # Save trained model
    joblib.dump(model, "outputs/iris_model.pkl")
    print("Saved: outputs/iris_model.pkl")

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Create heatmap
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Iris Decision Tree - Confusion Matrix")

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    train_model(args.test_size, args.random_state)