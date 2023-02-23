import pandas as pd

from sklearn.metrics import mean_squared_error


def missing_values(df):
    # todo; calculate the data availability for every column
    data_availability = df.count() / len(df)
    # todo; order columns by data availability
    ordered_columns = data_availability.sort_values().index
    df = df[ordered_columns]
    # todo; use the dataset mean and variance to find out missing value
    df.fillna(method='ffill', axis=0, inplace=True)
    return df


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
