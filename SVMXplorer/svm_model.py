from sklearn import svm

class SVM_Model:
    def __init__(self, kernel='linear', c=1, gamma='scale', degree=3, coef0=0):
        """
        Initialize the SVM model with specified parameters.

        Parameters:
            kernel (str): Kernel type for the SVM model. Default is 'linear'.
            c (float): Penalty parameter C of the error term. Default is 1.
            gamma (str): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. Default is 'scale'.
            degree (int): Degree of the polynomial kernel function. Default is 3.
            coef0 (float): Independent term in the polynomial kernel function. Default is 0.
        """
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
    
    def fit_model(self, x, y):
        """
        Fit the SVM model to the training data.

        Parameters:
            x (array-like, shape (n_samples, n_features)): Training data.
            y (array-like, shape (n_samples,)): Target values.

        Returns:
            None
        """
        self.clf = svm.SVC(C=self.c, kernel=self.kernel, degree=self.degree,
                           gamma=self.gamma, coef0=self.coef0)
        self.clf.fit(x, y)

    def predict_model(self, x):
        """
        Predict the target values for the given data.

        Parameters:
            x (array-like, shape (n_samples, n_features)): Data to predict.

        Returns:
            array-like, shape (n_samples,): Predicted target values.
        """
        return self.clf.predict(x)