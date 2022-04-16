from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class Builder:
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        assert(len(self.X_train) == len(self.Y_train))

    def get_subsample(self, df_share):
        """
        1. Copy train dataset
        2. Shuffle data (don't miss the connection between X_train and y_train)
        3. Return df_share %-subsample of X_train and y_train
        """
        assert(0 < df_share <= 100)
        (df_share / 100)
        divider = int((df_share/100)*len(self.X_train))
        X, Y = shuffle(self.X_train[:divider], self.Y_train[:divider])
        return X, Y


if __name__ == "__main__":
    """
    1. Load iris dataset
    2. Shuffle data and divide into train / test.
    """
    data = load_iris()
    print(f"Iris dataset: {data.data.shape[0]} items with {data.data.shape[1]} features")

    X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    pattern_item = Builder(X_train, Y_train)

    for df_share in range(10, 101, 10):
        curr_X_train, curr_y_train = pattern_item.get_subsample(df_share)
        """
        1. Preprocess curr_X_train, curr_y_train in the way you want
        2. Train Linear Regression on the subsample
        3. Save or print the score to check how df_share affects the quality
        """
        reg = LinearRegression().fit(curr_X_train, curr_y_train)
        print(f"{reg.score(X_test, Y_test):0.3f} score on {len(curr_X_train)} items")
