import numpy as np

import matplotlib.pyplot as plt


class Protein:

    def __init__(self, array):

        self.array = array.reshape(24, 82)

    def most_likely_sequence(self):
        return self.array.argmax(axis=0)

    def plot_sequence(self, title='', save='', show=True):

        plt.figure(figsize=(10, 10))
        plt.imshow(self.array, cmap='hot')
        plt.colorbar(orientation='horizontal', pad=.05, aspect=50)

        plt.title(title, size=15)

        if save:
            plt.savefig(save, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()
