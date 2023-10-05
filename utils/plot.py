import matplotlib.pyplot as plt
import numpy as np


def plot_loss(train, val, loss_name):
    plt.clf()
    plt.plot(np.arange(len(train)), train, 'r', alpha=0.8, label='train')  # c='#7B68EE'
    if len(val) != 0:
        plt.plot(np.arange(len(val)), val, c='#4169E1', label='val')
    plt.xlabel('Epochs')
    plt.ylabel(f'{loss_name}')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.draw()
    plt.gcf().canvas.flush_events()


def save_figure():
    return
