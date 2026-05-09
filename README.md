# Iris Classifier (Decision Tree)

## Overview

This project builds a machine learning model to classify iris flowers into three species: setosa, versicolor, and virginica.

The model is trained using a Decision Tree Classifier from scikit-learn and evaluated using accuracy and a confusion matrix.

---

## Dataset

The dataset is loaded from scikit-learn and contains four features:

* Sepal length
* Sepal width
* Petal length
* Petal width

---

## Model

A Decision Tree Classifier is trained on 80% of the data and tested on the remaining 20%.

---

## Results

* The model achieves high accuracy on the test set
* A confusion matrix is generated to evaluate performance
* The confusion matrix is saved in the `outputs/` folder

---

## Quick Start

Clone the repository and run the project:

```bash
git clone <your-repo-url>
cd iris-classifier

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python src/train.py
```

---

## Project Structure

```
iris-classifier/
├── data/
├── notebooks/
│   └── iris_model.ipynb
├── src/
│   └── train.py
├── tests/
├── outputs/
├── requirements.txt
├── README.md
└── .gitignore
```
