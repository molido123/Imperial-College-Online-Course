import pandas as pd
from sklearn.impute import KNNImputer

from sklearn.metrics import mean_squared_error


def missing_values(df):
    # todo; calculate the data availability for every column
    data_availability = df.count() / len(df)
    # todo; order columns by data availability
    ordered_columns = data_availability.sort_values().index
    df = df[ordered_columns]
    # todo; use the dataset mean and variance to find out missing value
    # todo; use a KNN to find out missing value
    # For beginners, you can use the sklearn knn data imputation implementation: sklearn.impute.KNNImputer
    # Advanced students can implement their own KNN data imputation pipeline
    imputer = KNNImputer()
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # todo; compare both strategies with the original dataset

    return df_imputed


def main():
    # if you prefer to work on normalization only, you can work with the original dataset
    df = pd.read_csv("../data/iris_numeric_dataset.missing.csv")
    df_original = pd.read_csv("../data/iris_numeric_dataset.original.csv")

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
