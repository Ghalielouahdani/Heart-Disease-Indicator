import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
            Call set_arguments function of this class.
        """

        self.k=k
        self.task_kind=task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.trainingData = training_data
        self.trainingLabels = training_labels
        pred_labels = self.predict(training_data)
        
        # For KNN, we don't need to return predictions during fit
        # as it's a lazy learner
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        #  Vectorised implementation using np.apply_along_axis 
        def _knn_single(x_query, training_features, training_labels, k, task_kind):
            """
            Helper function that returns the KNN prediction for one sample.
            This function will be applied to each test sample via np.apply_along_axis.
            """
            # Compute Euclidean distances to all training points
            dists = np.sqrt(np.sum((training_features - x_query) ** 2, axis=1))
            # Indices of the k nearest neighbours
            nn_idx = np.argpartition(dists, k)[:k]
            nn_labels = training_labels[nn_idx]

            if task_kind == "classification":
                return np.bincount(nn_labels.astype(int)).argmax()
            else:
                return np.mean(nn_labels)

        # Apply the helper across axis 1 (each row is a query sample)
        test_labels = np.apply_along_axis(
            _knn_single,
            axis=1,
            arr=test_data,
            training_features=self.trainingData,
            training_labels=self.trainingLabels,
            k=self.k,
            task_kind=self.task_kind
        )

        return test_labels