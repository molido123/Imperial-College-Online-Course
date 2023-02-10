import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, cohen_kappa_score


# compute the majority label
def compute_vote_agreement(row):
    count_agree = 0
    count_ela = 0
    for i in row:
        if str(i) == "agreement":
            ++count_agree
        elif str(i) == "elaboration":
            ++count_ela
    if count_agree > count_ela:
        return "agreement"
    else:
        return "elaboration"


# y is the ground truth labels, yp are the ones we compare with
def compute_confusion_matrix(y, yp):
    tn, fp, fn, tp = None, None, None, None
    # todo; implement this and return the correct values
    return tn, fp, fn, tp


# y is the ground truth labels, yp are the ones we compare with
def compute_recall(y, yp):
    # todo; implement this and return the correct values
    return 0.


# y is the ground truth labels, yp are the ones we compare with
def compute_precision(y, yp):
    # todo; implement this and return the correct values
    return 0.


# y is the ground truth labels, yp are the ones we compare with
def compute_f1(y, yp):
    # todo; implement this and return the correct values
    return 0.


# y is the ground truth labels, yp are the ones we compare with
def compute_accuracy(y, yp):
    # todo; implement this and return the correct values
    return 0.


# y is the ground truth labels, yp are the ones we compare with
def compute_cohen_kappa(y, yp):
    # todo; implement this and return the correct values
    return 0.


def compute_metrics(y, yp):
    stn, sfp, sfn, stp = confusion_matrix(y, yp).ravel()
    tn, fp, fn, tp = compute_confusion_matrix(y, yp)

    print(f"TP: {tp}. The value computed by Sklearn is: {stp}")
    print(f"FP: {fp}. The value computed by Sklearn is: {sfp}")
    print(f"TN: {tn}. The value computed by Sklearn is: {stn}")
    print(f"TP: {fn}. The value computed by Sklearn is: {sfn}")

    print(
        f"   Recall: {compute_recall(y, yp):.4} The value computed by Sklearn is: {recall_score(y, yp, average='macro'):.4}")
    print(
        f"Precision: {compute_precision(y, yp):.4} The value computed by Sklearn is: {precision_score(y, yp, average='macro'):.4}")
    print(f"       F1: {compute_f1(y, yp):.4}. The value computed by Sklearn is: {f1_score(y, yp, average='macro'):.4}")
    print(f" Accuracy: {compute_accuracy(y, yp):.4} The value computed by Sklearn is: {accuracy_score(y, yp):.4}")
    print(f"        K: {compute_cohen_kappa(y, yp):.4} The value computed by Sklearn is: {cohen_kappa_score(y, yp):.4}")


def main():
    df = pd.read_csv("data/coarse_discourse_dataset.simple.csv")

    # compute the agreed label via majority voting
    df["majority_label"] = df.apply(compute_vote_agreement, axis=1)

    print(df.head())
    print()

    print("--- Comparing annotator 1 vs annotator 2")
    compute_metrics(df.annot1.values, df.annot2.values)
    print()

    print("--- Comparing majority vs annotator 1")
    compute_metrics(df.majority_label.values, df.annot1.values)
    print()

    print("--- Comparing majority vs annotator 2")
    compute_metrics(df.majority_label.values, df.annot2.values)
    print()

    print("--- Comparing majority vs annotator 3")
    compute_metrics(df.majority_label.values, df.annot3.values)


if __name__ == "__main__":
    main()
