import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, append_bias_term


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=2000, reg_strength=0.001, weight_init="random", tol=1e-4, patience=10):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
            reg_strength (float): strength of L2 regularization
            weight_init (str): "zeros" or "random"
            tol (float): early‑stopping tolerance on relative loss improvement
            patience (int): number of consecutive checks with improvement < tol before stopping
        """
        self.lr = lr
        self.max_iters = max_iters
        self.reg_strength = reg_strength
        self.weight_init = weight_init
        self.tol = tol
        self.patience = patience

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # Append bias term to training data
        X = append_bias_term(training_data)
        N, D = X.shape

        # Get number of classes and convert labels to one-hot encoding
        n_classes = get_n_classes(training_labels)
        Y_onehot = label_to_onehot(training_labels, n_classes)

        # Initialize weights
        if self.weight_init == "random":
            rng = np.random.default_rng(seed=42)
            self.W = rng.normal(0, 0.01, size=(D, n_classes))
        else:
            self.W = np.zeros((D, n_classes))

        # Training loop with gradient descent
        prev_loss = np.inf
        patience_counter = 0
        for i in range(self.max_iters):
            # Compute scores and apply softmax
            scores = np.dot(X, self.W)
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Compute gradient of cross-entropy loss with L2 regularization
            grad = np.dot(X.T, (probs - Y_onehot)) / N
            grad += 2 * self.reg_strength * self.W  # L2 regularization gradient

            # Update weights
            self.W -= self.lr * grad
            
            # Compute loss
            cross_ent = -np.mean(np.sum(Y_onehot * np.log(probs + 1e-12), axis=1))
            l2_loss = self.reg_strength * np.sum(self.W ** 2)
            total_loss = cross_ent + l2_loss

            # Print every 100 iters
            if i % 100 == 0:
                print(f"iter {i}: loss = {total_loss:.4f}")

            # Early stopping check
            rel_improv = (prev_loss - total_loss) / (abs(prev_loss) + 1e-12)
            if rel_improv < self.tol:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at iter {i} (no significant improvement)")
                    break
            else:
                patience_counter = 0
            prev_loss = total_loss

        # Return predictions on training data
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # Append bias term to test data
        X = append_bias_term(test_data)

        # Compute scores and apply softmax
        scores = np.dot(X, self.W)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Convert probabilities to predicted labels
        pred_labels = onehot_to_label(probs)
        return pred_labels
