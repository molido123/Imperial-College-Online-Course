from abc import ABC
from random import sample

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, cohen_kappa_score
from sklearn.preprocessing import LabelBinarizer
from torch import optim
from torch.utils.data import IterableDataset, DataLoader


def label_vote(piece: list) -> int:
    tmp = max(piece)
    index = piece.index(tmp)
    return index


class IrisDataset(IterableDataset, ABC):
    def __init__(self, x, y, idx):
        super(IrisDataset).__init__()
        self._x = x
        self._y = y
        self._idx = idx

    def __iter__(self):
        return zip(self._x[self._idx], self._y[self._idx])

    def __len__(self):
        return len(self._x)


def load_datasets(batch_size):
    df = pd.read_csv("data/iris.csv")
    # getting the input features from the dataset
    x = np.float32(df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values)
    # The LabelBinarizer converts a discrete label (i.e. setosa) into a binary numeric representation [1, 0, 0],
    # where each column is different for different labels
    # todo; Advanced: How would you implement a different strategy? Would that change the cost function?
    y = np.float32(LabelBinarizer().fit_transform(df["species"].values))
    # select randomly 90% of the samples and place them into the training set
    train_idx = sample(range(len(x)), round(len(x) * 0.9))
    # the remaining samples (10%) will consist in the test set
    # todo; Advanced: how would you generated the validation set?
    test_idx = [x for x in range(len(x)) if x not in train_idx]
    train_ds = IrisDataset(x, y, train_idx)
    test_ds = IrisDataset(x, y, test_idx)
    return DataLoader(train_ds, batch_size=batch_size), DataLoader(test_ds, batch_size=batch_size)


def main():
    # the batch size is currently set to 4, but you can test with other batches to see if the results are changing
    train_dl, test_dl = load_datasets(batch_size=4)
    # Build a Sequential model, with a two dense linear level, a ReLU activation in the middle and a sigmoid
    # activation function at the end The input size has the match the number of features (4), while the output has to
    # match the encoding layer (3) The output of the first linear layer size has to match the input size of the
    # second layer. In this case it is 8.
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
        torch.nn.Sigmoid()
    )
    # What would be the most suitable loss function to use here?
    # Have a look at https://pytorch.org/docs/stable/nn.html#loss-functions and choose the right one.
    # Please note that the last activation function is already a sigmoid, so BCE Loss can be a good option.
    loss_fn = torch.nn.BCELoss()  # todo; replace this with a suitable loss function

    # Use the optim package to define an Optimizer that will update the weights of the model for us.
    # Have a look at https://pytorch.org/docs/stable/optim.html for more options
    # Adam is usually a safe option, and you have to pass model.parameters() to optimise correctly
    # lr is the efficiency of the training
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # todo; replace this with a suitable function

    # number of epochs to train on, you can check if this is decreasing
    num_epochs = 500
    avg_loss_list = []

    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in train_dl:
            # Before the backward pass, use the optimizer object to zero all the
            # gradients for the Tensors it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

        avg_loss = total_loss / len(train_dl)
        avg_loss_list.append(avg_loss)
        print(f"Epoch: {epoch}, Training loss: {avg_loss}")

    # todo; plot the average loss using matplotlib
    y = []
    for i in avg_loss_list:
        y.append(i.item())
    x = list(range(1, len(y) + 1))
    plt.plot(x, y)
    plt.savefig("result.png")
    plt.show()
    # todo; test the Precision, Recall, F1 of the model - you can use sklearn functions
    # these are the same functions you used in Tutorial 1
    real = numpy.empty([0, 3], dtype=float, order='C')
    pre = numpy.empty([0, 3], dtype=float, order='C')
    for x, y in test_dl:
        y_pred = model(x)
        real = np.insert(real, 0, values=y.numpy().copy(), axis=0)
        pre = np.insert(pre, 0, values=y_pred.detach().numpy().copy(), axis=0)
    yy = []
    pp = []
    for i in real:
        yy.append(label_vote(list(i)))
    for i in pre:
        pp.append(label_vote(list(i)))
    print(f"Recall:{recall_score(yy, pp, average='macro'):.4}")
    print(f"Precision: {precision_score(yy, pp, average='macro'):.4}")
    print(f"       F1: {f1_score(yy, pp, average='macro'):.4}")
    print(f" Accuracy:  {accuracy_score(yy, pp):.4}")
    print(f"        K: {cohen_kappa_score(yy, pp):.4}")


if __name__ == "__main__":
    main()
