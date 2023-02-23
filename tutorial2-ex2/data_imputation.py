import pandas as pd

from sklearn.metrics import mean_squared_error

"""
the solution is in the solution folder
"""


def missing_values(df):
    # todo; calculate the data availability for every column

    # todo; order columns by data availability

    # todo; use the dataset mean and variance to find out missing value

    # todo; use a KNN to find out missing value
    # For beginners, you can use the sklearn knn data imputation implementation: sklearn.impute.KNNImputer
    # Advanced students can implement their own KNN data imputation pipeline
    # the solution is in the solution folder
    # todo; compare both strategies with the original dataset
    """
        we can see if we use the KNN to estimate the missing value, we get the least MSE.
        because if we fill the nan with its corresponding column mean value, we will ignore the inner relationship
        between the variables.
    """
    return df


def main():
    # if you prefer to work on normalization only, you can work with the original dataset
    df = pd.read_csv("data/iris_numeric_dataset.missing.csv")
    df_original = pd.read_csv("data/iris_numeric_dataset.original.csv")

    # print the missing numbers
    print("Missing numbers:")
    print(df.isna().sum())
    print()

    df = missing_values(df)

    # calculate the Mean Squared Error between the predicted missing values and original data
    mse_error = df.fillna(0).apply(lambda column: mean_squared_error(df_original[column.name], column.values),
                                   axis=0).mean()
    print(f"Mean Squared Error between the missing data (corrected) and original is {mse_error}")


if __name__ == '__main__':
    main()
