from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

random_state = 42


class Facade:
    def __init__(self, x_train, y_train) -> None:
        global random_state
        self.X_train = x_train
        self.Y_train = y_train

        assert(len(self.X_train) == len(self.Y_train))


        self.predictors = [
            LinearRegression(),
            SVC(gamma='auto', random_state=random_state),
            RandomForestClassifier(max_depth=5, random_state=random_state),
            KNeighborsClassifier(7)
        ]

    def fit(self):
        """
        Fit classifiers from the initialization stage
        """
        for predictor in self.predictors:
            predictor.fit(self.X_train, self.Y_train)

    def predict(self, x):
        """
        Get predicts from all the classifiers and return
        the most popular answers
        """
        y = []
        for predictor in self.predictors:
            y.append(np.round(np.array(predictor.predict(x))))

        y_best, _ = np.unique(y, axis=0)
        '''
        print(y_best)
        y = np.array(y)
        y_best = []
        for i in range(y.shape[1]):
            values, _ = np.unique(y[:, i], return_counts=True)
            y_best.append(int(values[0]))

        print(y_best)
        '''
        return y_best


if __name__ == "__main__":
    """
    1. Load iris dataset
    2. Shuffle data and divide into train / test.
    3. Prepare classifiers to initialize <StructuralPatternName> class.
    4. Train the ensemble
    """
    data = load_iris()
    print(f"Iris dataset: {data.data.shape[0]} items with {data.data.shape[1]} features")

    X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=random_state, shuffle=True)
    my_super_estimator = Facade(X_train, Y_train)
    my_super_estimator.fit()
    preds = my_super_estimator.predict(X_test)
    #print(preds)
    print(f"My super estimator has {100*sum(preds == Y_test)/len(Y_test):0.1f}% accuracy")
