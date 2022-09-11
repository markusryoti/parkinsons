import os
import pandas as pd
import matplotlib.pyplot as plt
import torch


def get_files(dpath):
    X_train, X_test, y_train, y_test = [], [], [], []

    for root, dirs, files in os.walk(dpath):
        for name in files:
            file_path = os.path.join(root, name)
            cur_dir = os.path.dirname(file_path)
            label = cur_dir.split('/')[-1]
            train_test = cur_dir.split('/')[-2]

            if train_test == 'training':
                X_train.append(file_path)
                y_train.append(label)
            else:
                X_test.append(file_path)
                y_test.append(label)

    return X_train, X_test, y_train, y_test


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated
    # plt.show()


def load_model(fpath, model_class):
    loaded_model = model_class()
    loaded_model.load_state_dict(torch.load(fpath))
    loaded_model.eval()

    return loaded_model


def create_data(X, y):
    return pd.DataFrame({'filename': X, 'healthy': y})


def split_train_data(df):
    df = df.sample(frac=1)
    num_val = int(len(df) * 0.3)

    val = df.iloc[:num_val]
    train = df.iloc[num_val:]

    assert (len(val) + len(train) == len(df))

    return train, val
