from random import random

import pandas as pd

availability = {"sepal.length": 0.8, "sepal.width": 0.9, "petal.length": 0.95, "petal.width": 0.78, "variety": 0.7}


def main():
    df = pd.read_csv("data/iris_numeric_dataset.original.csv")
    for field, th in availability.items():
        df[field] = df[field].apply(lambda value: value if random() <= th else None)
    df.to_csv("data/iris_numeric_dataset.missing.csv", index=False)


if __name__ == '__main__':
    main()
