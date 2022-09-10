import os
import matplotlib.pyplot as plt


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
