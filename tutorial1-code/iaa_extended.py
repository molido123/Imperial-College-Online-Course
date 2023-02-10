import pandas as pd


def compute_confusion_matrix(y, yp):
    result = [[]]
    # todo; implement this and return the correct value
    return result


def display_confusion_matrix(m):
    # todo; implement this and return the correct value
    pass


def compute_recall_score(y, yp):
    # todo; implement this and return the correct value
    return None


def compute_precision_score(y, yp):
    # todo; implement this and return the correct value
    return None


def compute_f1_score(y, yp):
    # todo; implement this and return the correct value
    return None


def compute_accuracy_score(y, yp):
    # todo; implement this and return the correct value
    return None


def compute_cohen_kappa_score(y, yp):
    # todo; implement this and return the correct value
    return None


def compute_fleiss_kappa_score(df):
    # Fleiss' Kappa score is an extension of Cohen's Kappa for more than 2 annotators
    # todo; implement this and return the correct value

    return None


def compute_vote_agreement(row):
    result = "n/a"
    # todo; implement this and return the correct value
    return result


def compute_metrics(y, yp):
    m = compute_confusion_matrix(y, yp)
    display_confusion_matrix(m)

    print(f"   Recall: {compute_recall_score(y, yp):.4}")
    print(f"Precision: {compute_precision_score(y, yp):.4}")
    print(f"       F1: {compute_f1_score(y, yp):.4}")
    print(f" Accuracy: {compute_accuracy_score(y, yp):.4}")
    print(f"        K: {compute_cohen_kappa_score(y, yp):.4}")


def main():
    df = pd.read_csv("data/coarse_discourse_dataset.csv")

    # compute the agreed label via majority voting
    df["majority_label"] = df.apply(compute_vote_agreement, axis=1)

    print(df.head())
    print()

    print("--- Comparing annotator 1 vs annotator 2")
    compute_metrics(df.annot1.values, df.annot2.values)
    print("--- Comparing majority vs annotator 1")
    compute_metrics(df.majority_label.values, df.annot1.values)
    print("--- Comparing majority vs annotator 2")
    compute_metrics(df.majority_label.values, df.annot2.values)
    print("--- Comparing majority vs annotator 3")
    compute_metrics(df.majority_label.values, df.annot3.values)

    print(f" Fleiss Kappa: {compute_fleiss_kappa_score(df[['annot1', 'annot2', 'annot3']]):.4}")


if __name__ == "__main__":
    main()
