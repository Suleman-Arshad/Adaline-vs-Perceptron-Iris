# Perceptron vs. Delta Rule (Adaline) — Learning Dynamics on Iris

This project implements and compares two foundational neural network learning algorithms from scratch: the **Perceptron Learning Rule** and the **Gradient Descent Delta Rule (Adaline)**. Using the classic Iris dataset, the project visualizes how different activation functions and learning rates impact model convergence and error minimization.

## 🎯 Project Objectives
* **Scratch Implementation:** Manually coded weight initialization, net input calculation, and update rules using `NumPy`.
* **Algorithm Comparison:** Side-by-side analysis of the Perceptron (online learning) vs. the Delta Rule (batch gradient descent).
* **Activation Functions:** Experimentation with **Step**, **Linear**, and **Sigmoid** functions to observe their influence on the loss landscape.
* **Evaluation:** Implementation of an 80/20 train-test split with feature standardization to ensure fair performance metrics.
* **Hyperparameter Tuning:** Systematic sweep of learning rates ($\eta$) to identify the optimal rate for minimizing Mean Squared Error (MSE).

## 🧠 Core Concepts

### 1. Perceptron Learning Rule
The Perceptron is a "lazy" learner that only updates its weights when it makes a classification error. It uses a **Step Activation** function to produce hard labels (+1 or -1). Because the Setosa and Versicolor classes in the Iris dataset are linearly separable, the Perceptron converges to 100% accuracy almost instantly.

### 2. Gradient Descent Delta Rule
Unlike the Perceptron, the Delta Rule (Adaline) seeks to minimize the **Mean Squared Error (MSE)** through gradient descent. It uses a continuous activation function, allowing it to fine-tune the decision boundary even after achieving perfect classification. This project explores both **Linear** and **Sigmoid** variations of this rule.



## 📊 Key Results & Observations
The following visualization summarizes the training dynamics captured during execution:

![Learning Curves](<img width="2261" height="1457" alt="learning_curves" src="https://github.com/user-attachments/assets/db658081-b44d-4640-91ed-c30168070906" />
)

* **Convergence:** The Perceptron reaches zero misclassifications in just a few epochs, while the Delta Rule shows a smooth, exponential decay of error over time.
* **Learning Rate Sensitivity:** A learning rate of **$\eta = 0.1$** was found to be optimal, achieving the lowest MSE the fastest. A rate of **$0.001$** was too slow to converge within 100 epochs.
* **Stability:** Both algorithms achieved **100% accuracy** on the test set, confirming the robustness of the scratch implementations.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** NumPy (Math), Matplotlib (Visualization), Scikit-learn (Data Loading/Preprocessing)

## 🚀 How to Run
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/foundational-ml-scratch.git
    ```
2.  **Install Requirements:**
    ```bash
    pip install numpy matplotlib scikit-learn
    ```
3.  **Execute the Script:**
    ```bash
    python perceptron_vs_delta_rule.py
    ```

## 📝 License
This project is open-source and available under the **MIT License**.
