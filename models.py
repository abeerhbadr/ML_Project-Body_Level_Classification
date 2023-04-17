# logistic regression, svm ,linear regression, naive bayes, 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
# we have 4 classes in our dataset
class Model:
    def __init__(self, model_name, C=1.0, max_iter=1000):
        
        self.model_name = model_name
        if model_name == 'linear_regression':
            self.model = LinearRegression()
        elif model_name == 'logistic_regression':
            self.model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=max_iter, penalty='l2') # large max_iter to avoid convergence warning
        elif model_name == 'svm_linear':
            self.model = SVC(kernel='linear', C=C, random_state=0)
        elif model_name == 'svm_rbf':
            self.model = SVC(kernel='rbf', C=C, random_state=0)
        elif model_name == 'naive_bayes':
            self.model = GaussianNB()
        elif model_name == 'multinomial_naive_bayes':
            self.model = MultinomialNB()
        else:
            raise Exception('Invalid model name')


    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.model_name == 'linear_regression':
            y_pred = np.round(self.model.predict(X_test))
        else:
            y_pred = self.model.predict(X_test)
        return y_pred
    
    def evaluate(self, X_test, y_test, average='weighted', zero_division='warn'):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=zero_division)
        precision = precision_score(y_test, y_pred, average=average, zero_division=zero_division)
        recall = recall_score(y_test, y_pred, average=average, zero_division=zero_division)
        return accuracy, f1, precision, recall
