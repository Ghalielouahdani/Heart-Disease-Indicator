import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """

        self.k = k
        self.task_kind =task_kind

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
        test_labels = np.zeros(test_data.shape[0])

        for i in range(test_data.shape[0]):
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.trainingData - test_data[i])**2, axis=1))
            
            # Get k nearest neighbors
            nearest_neighbors = np.argsort(distances)[:self.k]
            neighbor_labels = self.trainingLabels[nearest_neighbors]
            
            if self.task_kind == "classification":
                test_labels[i] = np.bincount(neighbor_labels.astype(int)).argmax()
            else:
                test_labels[i] = np.mean(neighbor_labels)

        return test_labels