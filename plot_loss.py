import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def plot_full(train_losses, photo_losses, labels, sub_parts=2000):
    for train_loss, photo_loss, l in zip(train_losses, photo_losses, labels):
        train_loss_means = [train_loss[i:i + sub_parts] for i in
                            range(0, len(train_loss), sub_parts)]
        train_loss_means = [sum(loss) / len(loss) for loss in train_loss_means]

        photo_loss_means = [photo_loss[i:i + sub_parts] for i in
                            range(0, len(photo_loss), sub_parts)]
        photo_loss_means = [sum(loss) / len(loss) for loss in photo_loss_means]

        plt.plot(list(range(sub_parts, sub_parts * len(train_loss_means) + 1, sub_parts)), train_loss_means,
                 label=l + " train loss")
        plt.plot(list(range(sub_parts, sub_parts * len(photo_loss_means) + 1, sub_parts)), photo_loss_means,
                 label=l + " photo loss")

    plt.xlabel("batch")
    plt.ylabel("loss")
    plt.title(f"mean loss of {sub_parts} batches after each batch")
    plt.legend()
    plt.grid("on")
    plt.show()
    plt.clf()


def plot_epoch_loss(train_loss, validation_loss, labels):
    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = plt.subplot(111)

    max_range = 0
    for train_l, validation_l, l in zip(train_loss, validation_loss, labels):
        r = list(range(len(train_l)))
        max_range = max(len(r), max_range)
        ax.plot(r, train_l, label=l + " train")
        ax.plot(r, validation_l, label=l + " validation")
    plt.xlabel("epoch")
    plt.xticks(range(max_range))
    plt.ylabel("loss")
    plt.title("epoch train and validation loss")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fancybox=True, ncol=1, prop={'size': 6})
    ax.grid("on")
    plt.show()
    plt.clf()


def read_full_log(filepath):
    batch_train_loss = []
    batch_photo_loss = []
    batch_smooth_loss = []

    with open(str(filepath.joinpath("progress_log_full.csv"))) as f:
        reader = csv.reader(f, delimiter="\t", quotechar="\"")
        # skip header
        next(reader, None)
        for row in reader:
            train_loss, photo_loss, explainability_loss, smooth_loss = row

            batch_train_loss.append(float(train_loss))
            batch_photo_loss.append(float(photo_loss))
            batch_smooth_loss.append(float(smooth_loss))

    return batch_train_loss, batch_photo_loss


def read_summary_log(filepath):
    epoch_train_loss = []
    epoch_validation_loss = []

    with open(str(filepath.joinpath("progress_log_summary.csv"))) as f:
        reader = csv.reader(f, delimiter="\t", quotechar="\"")
        # skip header
        next(reader, None)
        for row in reader:
            train_loss, validation_loss = row

            epoch_train_loss.append(float(train_loss))
            epoch_validation_loss.append(float(validation_loss))

    return epoch_train_loss, epoch_validation_loss


if __name__ == "__main__":
    if len(sys.argv) > 1:
        labels = []

        train_losses = []
        photo_losses = []

        epoch_train_losses = []
        epoch_validation_losses = []

        for p in sys.argv[1:]:
            path = Path(p)

            model_name = ", ".join([parent.name for parent in list(path.parents)[:-2]])

            labels.append(model_name + ", " + path.name)

            losses = read_full_log(path)
            train_losses.append(losses[0])
            photo_losses.append(losses[1])

            plot_full([losses[0]], [losses[1]], [model_name + ", " + path.name])

            losses = read_summary_log(path)
            epoch_train_losses.append(losses[0])
            epoch_validation_losses.append(losses[1])

        plot_full(train_losses, photo_losses, labels)
        plot_epoch_loss(epoch_train_losses, epoch_validation_losses, labels)

        plot_epoch_loss([loss[:5] for loss in epoch_train_losses], [loss[:5] for loss in epoch_validation_losses],
                        labels)
    else:
        raise ValueError("No path given")
