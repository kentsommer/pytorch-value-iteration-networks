import numpy as np
import torch

def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out


def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep


def get_stats(loss, predictions, labels):
    cp = np.argmax(predictions.cpu().data.numpy(),1)
    error = np.mean(cp!=labels.cpu().data.numpy())
    return loss.data[0], error


def print_stats(epoch, avg_loss, avg_error, num_batches, time_duration):
    print(fmt_row(10, [epoch+1, avg_loss/num_batches, 
                    avg_error/num_batches, time_duration]))


def print_header():
    print(fmt_row(10, ["Epoch", "Train Loss", "Train Error", "Epoch Time"]))