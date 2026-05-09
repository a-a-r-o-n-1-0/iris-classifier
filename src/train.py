import argparse
import os

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

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)

    print("\nConfusion Matrix:")
    print(cm_df)

    # Save plot
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    plt.xticks(range(len(iris.target_names)), iris.target_names, rotation=45)
    plt.yticks(range(len(iris.target_names)), iris.target_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

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