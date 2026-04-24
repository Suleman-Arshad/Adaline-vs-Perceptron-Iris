# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fix the random seed so every run produces identical results.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ACTIVATION FUNCTIONS  (stand-alone so they can be reused / swapped)
def step_activation(net_input: np.ndarray) -> np.ndarray:
    # Hard-threshold (step) activation function — the classic Perceptron choice.
    return np.where(net_input >= 0.0, 1, -1)

def linear_activation(net_input: np.ndarray) -> np.ndarray:
    # Linear activation is the identity function — it returns the net input unchanged.
    return net_input


def sigmoid_activation(net_input: np.ndarray) -> np.ndarray:
   # Sigmoid activation squashes the net input into the (0, 1) range, making it suitable
    clipped = np.clip(net_input, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))

# BASE CLASS  (shared scaffolding for both algorithms)
class LinearClassifierBase:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None          # populated by fit()
        self.bias = None             # populated by fit()
        self.errors_per_epoch = []   # populated by fit()

    def _initialize_weights(self, number_of_features: int) -> None:
        # Initialize weights to small random values from a normal distribution.
        self.weights = np.random.normal(loc=0.0, scale=0.01,
                                        size=number_of_features)
        self.bias = 0.0

    def net_input(self, X: np.ndarray) -> np.ndarray:
        # Compute the net input (weighted sum + bias) for each sample in X.
        return np.dot(X, self.weights) + self.bias

# PERCEPTRON CLASS
class Perceptron(LinearClassifierBase):
    # The Perceptron is the original linear classifier that uses the step activation
    def __init__(self, learning_rate: float = 0.1, epochs: int = 50):
        # Delegate to the base class constructor.
        super().__init__(learning_rate=learning_rate, epochs=epochs)

    def activation(self, net_input: np.ndarray) -> np.ndarray:
       # For the Perceptron, the activation function is the step function, which outputs hard class labels directly.
        return step_activation(net_input)

    def predict(self, X: np.ndarray) -> np.ndarray:
       # The predict method applies the net input and then the activation function to produce final class labels.
        return self.activation(self.net_input(X))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Perceptron":
       # Train the Perceptron using the online learning rule (per-sample updates).
        self._initialize_weights(number_of_features=X_train.shape[1])
        self.errors_per_epoch = []

        for epoch in range(self.epochs):
            number_of_misclassifications = 0

            # online update: process one sample at a time
            for x_i, y_i in zip(X_train, y_train):
                # make a prediction using the current weights
                prediction = self.predict(x_i.reshape(1, -1))[0]

                # compute the error signal
                error = y_i - prediction   # 0 if correct, ±2 if wrong

                # update weights and bias
                #   The update magnitude is η * error * x_i.
                #   If error == 0, this is a no-op, which is efficient and clean.
                self.weights += self.learning_rate * error * x_i
                self.bias    += self.learning_rate * error

                # Count misclassifications (error != 0 means a mistake was made)
                if error != 0:
                    number_of_misclassifications += 1

            # Store this epoch's count so we can plot convergence later
            self.errors_per_epoch.append(number_of_misclassifications)

        return self  # return self to allow: model = Perceptron().fit(X, y)

# DELTA RULE (ADALINE) CLASS
class DeltaRule(LinearClassifierBase):
   # The Delta Rule (Adaline) is a linear classifier that uses the mean-squared error loss and can work with different activation functions.
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 100,
        activation_fn=linear_activation,   # <-- easily swappable!
    ):
        super().__init__(learning_rate=learning_rate, epochs=epochs)
        # Store the chosen activation function for use in both training and prediction.
        self._activation_fn = activation_fn

    def activation(self, net_input: np.ndarray) -> np.ndarray:
       # The activation function is applied to the net input to produce the model's output. For the Delta Rule, this can be linear (identity) or sigmoid, depending on how we instantiate it.
        return self._activation_fn(net_input)

    def predict(self, X: np.ndarray) -> np.ndarray:
      # To make predictions, we compute the continuous output from the activation function and then apply a threshold to convert it into class labels (+1 or -1).
        continuous_output = self.activation(self.net_input(X))

        # The thresholding depends on the activation function:
        # - For sigmoid activation, we threshold at 0.5 (the midpoint of the sigmoid curve).
        # - For linear activation, we threshold at 0.0 (the decision boundary).
        if self._activation_fn is sigmoid_activation:
            return np.where(continuous_output >= 0.5, 1, -1)
        else:
            return np.where(continuous_output >= 0.0, 1, -1)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DeltaRule":
        # Train the Delta Rule using batch gradient descent to minimize mean-squared error.
        n_samples = X_train.shape[0]
        self._initialize_weights(number_of_features=X_train.shape[1])
        self.errors_per_epoch = []

        for epoch in range(self.epochs):
            # forward pass (all samples at once — batch mode)
            net = self.net_input(X_train)           # shape: (n_samples,)
            output = self.activation(net)           # shape: (n_samples,)
            errors = y_train - output               # shape: (n_samples,)

            # compute mean-squared error (our loss function)
            # MSE = (1/N) Σ (error_i^2)  →  we divide by 2N for a cleaner gradient expression
            mse = (1.0 / (2.0 * n_samples)) * np.sum(errors ** 2)
            self.errors_per_epoch.append(mse)

            # gradient descent weight update
            # The weight update is derived from the MSE loss gradient:
            #   ∂MSE/∂w = -(1/N) X^T
            weight_gradient = (1.0 / n_samples) * X_train.T.dot(errors)
            bias_gradient   = (1.0 / n_samples) * errors.sum()

            self.weights += self.learning_rate * weight_gradient
            self.bias    += self.learning_rate * bias_gradient

        return self

# UTILITY FUNCTIONS
def compute_accuracy(model, X: np.ndarray, y_true: np.ndarray) -> float:
    y_predicted = model.predict(X)
    correct = np.sum(y_predicted == y_true)
    return (correct / len(y_true)) * 100.0


def print_results(model_name: str,
                  model,
                  X_train, y_train,
                  X_test, y_test) -> None:
    train_acc = compute_accuracy(model, X_train, y_train)
    test_acc  = compute_accuracy(model, X_test,  y_test)
    bar = "─" * 45
    print(f"\n{bar}")
    print(f"  {model_name}")
    print(bar)
    print(f"  Train Accuracy : {train_acc:.2f}%")
    print(f"  Test  Accuracy : {test_acc:.2f}%")
    print(bar)

# DATA PREPARATION
def load_and_prepare_data():
    print("\n" + "=" * 60)
    print("  STEP 1 — Data Preparation")
    print("=" * 60)

    # Load raw data
    iris = load_iris()
    X_full = iris.data      # shape: (150, 4)
    y_full = iris.target    # 0=Setosa, 1=Versicolor, 2=Virginica

    # We will focus on a binary classification task (Setosa vs Versicolor) to keep things simple and allow both algorithms to shine. This means we will filter out the Virginica class (label 2).
    binary_mask  = y_full != 2               # True for classes 0 and 1
    X_binary     = X_full[binary_mask]       # shape: (100, 4)
    y_raw        = y_full[binary_mask]       # values: 0 or 1

   # Convert labels to -1 and +1 for our algorithms (Perceptron and Delta Rule expect this format).
    y_binary = np.where(y_raw == 0, -1, 1)  # shape: (100,)

    print(f"  Total samples after filtering : {X_binary.shape[0]}")
    print(f"  Features                      : {X_binary.shape[1]}")
    print(f"  Classes                       : Setosa (-1) vs Versicolor (+1)")

   # Scale features to zero mean and unit variance — important for gradient descent convergence.
    scaler  = StandardScaler()

   # Split into train and test sets (80% train, 20% test) with stratification to maintain class balance.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_binary, y_binary,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_binary   # keep class ratios equal in both splits
    )

    X_train = scaler.fit_transform(X_train_raw)   # fit + transform on train
    X_test  = scaler.transform(X_test_raw)        # only transform on test

    print(f"  Training samples              : {X_train.shape[0]}")
    print(f"  Testing  samples              : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test

# TRAINING & EVALUATION
def train_and_evaluate(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("  STEP 2 — Training & Evaluation")
    print("=" * 60)

    # Perceptron
    print("\n  Training Perceptron …")
    perceptron = Perceptron(learning_rate=0.1, epochs=50)
    perceptron.fit(X_train, y_train)
    print_results("Perceptron  (η=0.10, 50 epochs)",
                  perceptron, X_train, y_train, X_test, y_test)

    # Delta Rule with linear activation (classic Adaline)
    print("\n  Training Delta Rule (linear activation) …")
    delta_linear = DeltaRule(learning_rate=0.01,
                             epochs=100,
                             activation_fn=linear_activation)
    delta_linear.fit(X_train, y_train)
    print_results("Delta Rule  (linear, η=0.01, 100 epochs)",
                  delta_linear, X_train, y_train, X_test, y_test)

    # Delta Rule with sigmoid activation (logistic regression style)
    print("\n  Training Delta Rule (sigmoid activation) …")
    delta_sigmoid = DeltaRule(learning_rate=0.1,
                              epochs=100,
                              activation_fn=sigmoid_activation)
    delta_sigmoid.fit(X_train, y_train)
    print_results("Delta Rule  (sigmoid, η=0.10, 100 epochs)",
                  delta_sigmoid, X_train, y_train, X_test, y_test)

    return perceptron, delta_linear, delta_sigmoid

# HYPERPARAMETER TUNING — Learning Rate Comparison
def compare_learning_rates(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("  STEP 3 — Hyperparameter Tuning (Learning Rate)")
    print("=" * 60)

    learning_rates = [0.1, 0.01, 0.001]
    tuned_models = []

    for lr in learning_rates:
        model = DeltaRule(learning_rate=lr,
                          epochs=100,
                          activation_fn=linear_activation)
        model.fit(X_train, y_train)
        label = f"η = {lr}"
        tuned_models.append((label, model))

        train_acc = compute_accuracy(model, X_train, y_train)
        test_acc  = compute_accuracy(model, X_test,  y_test)
        print(f"  {label:<10}  →  Train: {train_acc:.1f}%   Test: {test_acc:.1f}%")

    return tuned_models

# VISUALISATION
def plot_all_results(perceptron, delta_linear, delta_sigmoid, tuned_models):
    plt.style.use("dark_background")
    ACCENT_BLUE    = "#4FC3F7"
    ACCENT_ORANGE  = "#FFB74D"
    ACCENT_GREEN   = "#81C784"
    ACCENT_PINK    = "#F48FB1"
    ACCENT_PURPLE  = "#CE93D8"
    ACCENT_YELLOW  = "#FFF176"
    GRID_COLOR     = "#2a2a2a"

    fig = plt.figure(figsize=(18, 10), facecolor="#0d0d0d")
    fig.suptitle(
        "Perceptron vs. Delta Rule (Adaline) — Learning Dynamics on Iris",
        fontsize=18, fontweight="bold", color="white", y=0.98
    )

    outer_grid = gridspec.GridSpec(2, 1, figure=fig,
                                   hspace=0.45,
                                   top=0.92, bottom=0.07)
    top_row    = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[0],
                                                  wspace=0.35)
    bottom_row = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[1],
                                                  wspace=0.35)

    #  Helper: shared axis styling 
    def style_ax(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=12, color="white", pad=10)
        ax.set_xlabel(xlabel, fontsize=10, color="#aaaaaa")
        ax.set_ylabel(ylabel, fontsize=10, color="#aaaaaa")
        ax.tick_params(colors="#aaaaaa")
        ax.set_facecolor("#111111")
        ax.grid(True, color=GRID_COLOR, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    #  Helper: draw a single curve with fill + final-value annotation 
    def draw_curve(ax, values, color, label, linestyle="-"):
        epochs_axis = range(1, len(values) + 1)
        ax.plot(epochs_axis, values, color=color, linewidth=2.2,
                linestyle=linestyle, label=label)
        ax.fill_between(epochs_axis, values, alpha=0.12, color=color)
        # Annotate the final value
        final_val = values[-1]
        ax.annotate(
            f"{final_val:.3f}",
            xy=(len(values), final_val),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color=color
        )

    #  Panel 1: Perceptron misclassifications 
    ax1 = fig.add_subplot(top_row[0])
    style_ax(ax1,
             title="Perceptron — Misclassifications per Epoch",
             xlabel="Epoch",
             ylabel="# Misclassified Samples")
    draw_curve(ax1, perceptron.errors_per_epoch, ACCENT_BLUE, "Perceptron")
    ax1.legend(fontsize=9, framealpha=0.3)

    #  Panel 2: Delta Rule (linear) MSE 
    ax2 = fig.add_subplot(top_row[1])
    style_ax(ax2,
             title="Delta Rule (Linear) — MSE per Epoch",
             xlabel="Epoch",
             ylabel="Mean Squared Error")
    draw_curve(ax2, delta_linear.errors_per_epoch, ACCENT_ORANGE, "Linear Adaline")
    ax2.legend(fontsize=9, framealpha=0.3)

    #  Panel 3: Delta Rule (sigmoid) MSE 
    ax3 = fig.add_subplot(top_row[2])
    style_ax(ax3,
             title="Delta Rule (Sigmoid) — MSE per Epoch",
             xlabel="Epoch",
             ylabel="Mean Squared Error")
    draw_curve(ax3, delta_sigmoid.errors_per_epoch, ACCENT_GREEN, "Sigmoid Adaline")
    ax3.legend(fontsize=9, framealpha=0.3)

    #  Panel 4: Learning rate comparison (loss curves) 
    lr_colors = [ACCENT_PINK, ACCENT_PURPLE, ACCENT_YELLOW]
    ax4 = fig.add_subplot(bottom_row[0])
    style_ax(ax4,
             title="Delta Rule — Effect of Learning Rate on MSE",
             xlabel="Epoch",
             ylabel="Mean Squared Error")
    for (label, model), color in zip(tuned_models, lr_colors):
        draw_curve(ax4, model.errors_per_epoch, color, label)
    ax4.legend(fontsize=9, framealpha=0.3)

    # Panel 5: Learning rate comparison (bar chart — final accuracy)
    ax5 = fig.add_subplot(bottom_row[1])
    style_ax(ax5,
             title="Delta Rule — Final MSE by Learning Rate",
             xlabel="Learning Rate",
             ylabel="Final MSE (last epoch)")
    lr_labels  = [lbl  for (lbl, _)   in tuned_models]
    final_mses = [mdl.errors_per_epoch[-1] for (_, mdl) in tuned_models]
    bars = ax5.bar(lr_labels, final_mses, color=lr_colors,
                   edgecolor="#333333", linewidth=0.8)
    for bar, val in zip(bars, final_mses):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(final_mses) * 0.01,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontsize=9, color="white")

    # Save the figure to a file before showing it (to preserve the dark background and high resolution).
    output_path = "learning_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Plot saved to: {output_path}")
    plt.show()

# MAIN ENTRY POINT
def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║   Perceptron vs. Delta Rule — From-Scratch Comparison    ║")
    print("╚" + "═" * 58 + "╝")

    # Step 1 ─ data preparation
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Step 2 ─ train & evaluate
    perceptron, delta_linear, delta_sigmoid = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # Step 3 ─ hyperparameter tuning
    tuned_models = compare_learning_rates(X_train, X_test, y_train, y_test)

    # Step 4 ─ visualise
    print("\n" + "=" * 60)
    print("  STEP 4 — Visualisation")
    print("=" * 60)
    plot_all_results(perceptron, delta_linear, delta_sigmoid, tuned_models)
    print("\n  ✓  All done! Review the plot to see convergence in action.")
    print("  ─" * 30 + "\n")

if __name__ == "__main__":
    main()