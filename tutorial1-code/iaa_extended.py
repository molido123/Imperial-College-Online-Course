import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)


def compute_confusion_matrix(y, yp):
    # todo; implement this and return the correct value
    categories = y.copy()
    list(categories).append(list(yp))
    categories = list(set(categories))
    categories.sort()
    result = np.zeros((len(categories), len(categories)))
    result = result.astype("int64")
    for i in range(len(categories)):
        row = y[yp == categories[i]]
        for j in range(len(categories)):
            result[i][j] = len(row[row == categories[j]])
    return result


def display_confusion_matrix(m):
    # todo; implement this and return the correct value
    print(m)
    pass


def compute_recall_score(y, yp):
    # todo; implement this and return the correct value
    # todo; implement this and return the correct values
    data = compute_confusion_matrix(y, yp)
    result = 0.
    for i in range(len(data[0])):
        result += data[i][i] / sum(data[i])
    return result / len(data[0])


def compute_precision_score(y, yp):
    # todo; implement this and return the correct value
    data = compute_confusion_matrix(y, yp)
    sum_data = sum(sum(data))
    result = 0.
    for i in range(len(data[0])):
        TP = data[i][i]
        FN = sum(data[i]) - TP
        sum_clo = 0
        for j in range(len(data[0])):
            sum_clo += data[j][i]
        FP = sum_clo - FN
        TN = sum_data - FP - FN - TP
        result += TP / (TP + FP)
    return result / len(data[0])


def compute_f1_score(y, yp):
    # todo; implement this and return the correct value
    # todo; implement this and return the correct value
    data = compute_confusion_matrix(y, yp)
    sum_data = sum(sum(data))
    result = 0.
    for i in range(len(data[0])):
        TP = data[i][i]
        FN = sum(data[i]) - TP
        sum_clo = 0
        for j in range(len(data[0])):
            sum_clo += data[j][i]
        FP = sum_clo - FN
        TN = sum_data - FP - FN - TP
        precision_i = TP / (TP + FP)
        recall_i = TP / (TP + FN)
        result += 2 * (precision_i * recall_i) / (precision_i + recall_i)
    return result / len(data[0])


def compute_accuracy_score(y, yp):
    # todo; implement this and return the correct value
    data = compute_confusion_matrix(y, yp)
    sum_data = sum(sum(data))
    result = 0.
    for i in range(len(data[0])):
        TP = data[i][i]
        FN = sum(data[i]) - TP
        sum_clo = 0
        for j in range(len(data[0])):
            sum_clo += data[j][i]
        FP = sum_clo - FN
        TN = sum_data - FP - FN - TP
        result += (TP + TN) / sum_data
    return result / len(data[0])


def compute_cohen_kappa_score(y, yp):
    # todo; implement this and return the correct value
    data = compute_confusion_matrix(y, yp)
    sum_data = sum(sum(data))
    sum_TP = 0
    for i in range(len(data[0])):
        sum_TP += data[i][i]
    p0 = sum_TP / sum_data
    pe = 0
    for i in range(len(data[0])):
        TP = data[i][i]
        FN = sum(data[i]) - TP
        sum_clo = 0
        for j in range(len(data[0])):
            sum_clo += data[j][i]
        FP = sum_clo - FN
        TN = sum_data - FP - FN - TP
        pe += (TP + FP) * (TP + FN) / (sum_data * sum_data)
    return (p0 - pe) / (1 - pe)


def compute_fleiss_kappa_score(df):
    # Fleiss' Kappa score is an extension of Cohen's Kappa for more than 2 annotators
    # todo; implement this and return the correct value
    categories = list(set(list(df['annot1'].values)))
    categories.sort()

    data1 = np.array(df['annot1'].values)
    sum_1 = len(data1)
    data2 = np.array(df['annot2'].values)
    sum_2 = len(data2)
    data3 = np.array(df['annot3'].values)
    sum_3 = len(data3)
    pj = []
    ni = []
    for i in categories:
        p1 = len(data1[data1 == i]) / sum_1
        p2 = len(data2[data2 == i]) / sum_2
        p3 = len(data3[data3 == i]) / sum_3
        pj.append((p1 + p2 + p3) / 3)
        ni.append(len(data1[data1 == i]) + len(data2[data2 == i]) + len(data3[data3 == i]))
    pij = sum(pj) / len(pj)
    e = 0
    for i in ni:
        e += (i / (len(ni) * (sum_3 + sum_2 + sum_1))) ** 2
    return (pij - e) / (1 - e)

    return None


def compute_vote_agreement(row):
    # todo; implement this and return the correct value
    data = {}
    for i in range(1, 4):
        if data.get(row[i]) is None:
            data[row[i]] = 0
        else:
            data[row[i]] = data[row[i]] + 1
    return max(data.items(), key=lambda x: x[1])[0]


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
