import numpy as np
import matplotlib.pyplot as plt
import imageio

# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is in charge of presenting the position and cluster label of a set of particles whose
# position has been discretized into a set of planes of constant Z. This is done with a GIF animation, plotting the resulting
# cluster assignments of each two-dimensional domain separatedly.
# Álvaro Tomás Gil - UIUC 2020


class clusterPlot:
    """This script is in charge of presenting the position and cluster label of a set of particles whose
    position has been discretized into a set of planes of constant Z. This is done with a GIF animation, plotting the resulting
    cluster assignments of each two-dimensional domain separatedly.
    For the input:
    data: 2D array of 3D particle positions, sorted into parallel planes of constant Z
    labels: list of equal length as data, assigning a cluster label applicable within each 2D domain. A label
    of -1 corresponds to a void particle.
    folder: folder in which to save resulting plots
    minClusters: minimum number of cluster labels with which to separate exclusively into void and cluster particles, and
    not within different assigned clusters"""
    def __init__(self, data, labels, folder='', minClusters=3000):
        self.labels = labels
        self.data = data
        self.folder = folder
        if len(np.unique(np.array(labels))) < minClusters:
            self.polyChrome = True
        else:
            self.polyChrome = False

        self.threedee = False
        if np.size(self.data, 1) == 3:
            self.zs = np.unique(self.data[:, 2])
            self.threedee = True
            if len(self.zs) == 1:
                self.data = self.data[:, :2]
                self.threedee = False

        self.unique_labels = np.unique(np.array(labels))
        strength = np.linspace(0, 0.8, len(self.unique_labels))
        np.random.shuffle(strength)
        self.colors = [plt.cm.nipy_spectral(each) for each in strength]
        np.random.shuffle(strength)
        self.colorsB = [plt.cm.nipy_spectral(each) for each in strength]

    def plotAll(self, title):
        if self.threedee:
            dz = abs(self.zs[1] - self.zs[0]) / 4

            def update(choose):
                fig, ax = plt.subplots()
                fig.set_size_inches(18.5, 9.5)

                relevant = ((self.data[:, 2] <= choose + dz) & (self.data[:, 2] >= choose - dz))
                self.plotInstance(self.data[relevant, :2],
                                  [self.labels[i] for i in range(len(self.labels)) if relevant[i]], ax)

                ax.set_title('Z = ' + '{0:03f}'.format(choose), fontsize=24)
                ax.set_xlabel('x [m]', fontsize=18)
                ax.set_ylabel('y [m]', fontsize=18)
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.set_xlim(np.min(self.data[:, 0]), np.max(self.data[:, 0]))
                ax.set_ylim(np.min(self.data[:, 1]), np.max(self.data[:, 1]))
                plt.axis('equal')

                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close()
                return image

            kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
            imageio.mimsave(self.folder + title + '.gif', [update(i) for i in self.zs], fps=2)
        else:
            fig, ax = plt.subplots()
            fig.set_size_inches(18.5, 9.5)
            ax.set_title(title)
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            plt.axis('equal')
            self.plotInstance(self.data, self.labels, ax)

    def plotInstance(self, data, labels, ax):
        if self.polyChrome:

            for k, col, colB in zip(self.unique_labels, self.colors, self.colorsB):
                size = 3
                if k == -1:
                    # Black used for noise.
                    col = [1, 0, 0]
                    size = 1

                class_member_mask = (labels == k)
                xy = data[class_member_mask]
                if len(xy) > 0:
                    ax.scatter(xy[:, 0], xy[:, 1], c=np.reshape(np.array(col), (1, -1)),
                               edgecolors=np.reshape(np.array(colB), (1, -1)), s=30, label='Cluster ' + str(k))
            ax.legend()
        else:
            s = 0.8 + (labels != -1) * 0.8
            col = {1: 'blue', 0: 'grey'}
            c = [col[d] for d in (labels != -1)]
            ax.scatter(data[:, 0], data[:, 1], s=s, c=c)

    def sizeHistogram(self, title):
        """This method plots a histogram of the number of particles contained within each DBSCAN cluster"""
        unique, counts = np.unique(np.array(self.labels), return_counts=True)
        plt.figure()
        plt.hist(counts, bins=100, density=True)
        plt.title(title)
        plt.xlabel('Particles per Cluster [-]')
        plt.ylabel('Fraction of Clusters [-]')
