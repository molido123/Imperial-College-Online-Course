import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, cohen_kappa_score


# compute the majority label
def compute_vote_agreement(row):
    count_agree = 0
    count_ela = 0
    for i in row:
        if str(i) == "agreement":
            count_agree = count_agree + 1
        elif str(i) == "elaboration":
            count_ela = count_ela + 1
    if count_agree > count_ela:
        return "agreement"
    else:
        return "elaboration"


# y is the ground truth labels, yp are the ones we compare with
def compute_confusion_matrix(y, yp):
    tn, fp, fn, tp = 0, 0, 0, 0
    # todo; implement this and return the correct values
    for i in range(len(y)):
        if yp[i] == "elaboration":
            if y[i] == yp[i]:
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if y[i] == yp[i]:
                tn = tn + 1
            else:
                fn = fn + 1

    return tn, fp, fn, tp


# y is the ground truth labels, yp are the ones we compare with
def compute_recall(y, yp):
    # todo; implement this and return the correct values
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    return tp / (tp + fn)


# y is the ground truth labels, yp are the ones we compare with
def compute_precision(y, yp):
    # todo; implement this and return the correct values
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    return (tn + tp) / len(y)


# y is the ground truth labels, yp are the ones we compare with
def compute_f1(y, yp):
    # todo; implement this and return the correct values
    recall = compute_recall(y, yp)
    precision = compute_precision(y, yp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# y is the ground truth labels, yp are the ones we compare with
def compute_accuracy(y, yp):
    # todo; implement this and return the correct values
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return accuracy


# y is the ground truth labels, yp are the ones we compare with
def compute_cohen_kappa(y, yp):
    # todo; implement this and return the correct values
    kappa = cohen_kappa_score(y, yp)
    return kappa


def compute_metrics(y, yp):
    stn, sfp, sfn, stp = confusion_matrix(y, yp).ravel()

    tn, fp, fn, tp = compute_confusion_matrix(y, yp)
    print(f"TP: {tp}. The value computed by Sklearn is: {stp}")
    print(f"FP: {fp}. The value computed by Sklearn is: {sfp}")
    print(f"TN: {tn}. The value computed by Sklearn is: {stn}")
    print(f"FN: {fn}. The value computed by Sklearn is: {sfn}")
    print(
        f"   Recall: {compute_recall(y, yp):.4} The value computed by Sklearn is: {recall_score(y, yp, average='macro'):.4}")
    print(
        f"Precision: {compute_precision(y, yp):.4} The value computed by Sklearn is: {precision_score(y, yp, average='macro'):.4}")
    print(f"       F1: {compute_f1(y, yp):.4}. The value computed by Sklearn is: {f1_score(y, yp, average='macro'):.4}")
    print(f" Accuracy: {compute_accuracy(y, yp):.4} The value computed by Sklearn is: {accuracy_score(y, yp):.4}")
    print(f"        K: {compute_cohen_kappa(y, yp):.4} The value computed by Sklearn is: {cohen_kappa_score(y, yp):.4}")
    print()


def main():
    df = pd.read_csv("data/coarse_discourse_dataset.simple.csv")

    # compute the agreed label via majority voting
    df["majority_label"] = df.apply(compute_vote_agreement, axis=1)

    print(df.head())

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
