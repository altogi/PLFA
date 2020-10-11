import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import imageio


# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is in charge of plotting by means of a GIF file, how a set of classified clusters evolve
# between two time instants. In order to do so, for every snapshot in Z, the cluster boundaries and skeleton points are plotted
# together, and the pairing between skeleton points is also displayed.
# Álvaro Tomás Gil - UIUC 2020

class temporalPlot:
    """This class is in charge of plotting by means of a GIF file, how a set of classified clusters evolve
    between two time instants. In order to do so, for every snapshot in Z, the cluster boundaries and skeleton points are plotted
    together, and the pairing between skeleton points is also displayed.
    For the inputs:
    skel: List of two elements, where each element is a 2D array of 3D positions of skeleton points for each of the time
    instants compared.
    bound: List of two elements, where each element is a 2D array of 3D positions of cluster boundary points for each of the time
    instants compared.
    labels: List of two elements, where each element is a list of cluster label for the skeleton points of each of the time
    instants compared.
    connections: 2D array of 3D points defining connections between skeleton points of the first time frame with the second one
    folder: folder in which to save resulting plots"""
    def __init__(self, skel, bound, labels, connections, folder=''):
        self.skel = skel
        self.bound = bound
        self.labels = labels
        self.folder = folder
        self.connections = connections

        self.zs = np.unique(skel[0][:, 2])

        # Combinations of marker and marker boundary colors are made in order to increase the possibilities of marker types
        self.skelUniqueLabels = [0, 0]
        self.skelColors = [0, 0]
        self.skelColorsB = [0, 0]
        self.boundC = [0, 0]
        self.cm = [cm.get_cmap('winter'), cm.get_cmap('autumn')]
        for i in range(2):
            self.skelUniqueLabels[i] = np.unique(self.labels[i])
            strength = np.linspace(0, 0.8, len(self.skelUniqueLabels[i]))
            np.random.shuffle(strength)
            self.skelColors[i] = [self.cm[i](each) for each in strength]
            np.random.shuffle(strength)
            self.skelColorsB[i] = [self.cm[i](each) for each in strength]
            normalized = (self.bound[i][:, 2] - np.min(self.zs)) / (np.ptp(self.zs))
            self.boundC[i] = self.cm[i](normalized)


    def snapPlot(self, title='Temporal Tracking of Particle Clusters'):
        """This method plots each two-dimensional domain separately, but joins all of them in a GIF animation which allows
        a 3D evolution of them to be visualized."""

        def update(choose):
            fig, ax = plt.subplots()
            fig.set_size_inches(18.5, 9.5)

            self.plotInstance(choose, ax)
            ax.legend()
            ax.set_title('Z = ' + '{0:03f}'.format(choose), fontsize=24)
            ax.set_xlabel('x [m]', fontsize=18)
            ax.set_ylabel('y [m]', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=15)
            plt.axis('equal')
            ax.set_xlim(np.min(self.bound[0][:, 0]), np.max(self.bound[0][:, 0]))
            ax.set_ylim(np.min(self.bound[0][:, 1]), np.max(self.bound[0][:, 1]))

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

        kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
        imageio.mimsave(self.folder + title + '.gif', [update(i) for i in self.zs], fps=2)

    def plotInstance(self, choose, ax):
        if len(self.zs) >= 2:
            dz = (self.zs[1] - self.zs[0]) / 4
        else:
            dz = 0.001 * self.zs[0]

        takeC = self.connections[:, 2] == choose
        ax.scatter(self.connections[takeC, 0], self.connections[takeC, 1], s=5, marker='D')

        for i in range(2):
            if i == 0:
                label = 'Old Skeleton - Cluster '
                labelB = 'Old Cluster Boundary'
            else:
                label = 'New Skeleton - Cluster '
                labelB = 'New Cluster Boundary'
            takeB = self.bound[i][:, 2] == choose
            takeS = ((self.skel[i][:, 2] >= choose - dz) & (self.skel[i][:, 2] <= choose + dz))
            self.plotLabels(ax, self.skel[i][takeS, :], self.labels[i][takeS], self.skelColors[i], self.skelColorsB[i], label, marker='P')
            ax.scatter(self.bound[i][takeB, 0], self.bound[i][takeB, 1], c=self.boundC[i][takeB], s=1, label=labelB)


    @staticmethod
    def plotLabels(ax, data, labels, colors, colorsB, label, marker='.', size=75):
        unique_labels = np.unique(labels)

        for k, col, colB in zip(unique_labels, colors, colorsB):
            class_member_mask = (labels == k)
            xy = data[class_member_mask]
            if len(xy) > 0:
                if label is not None:
                    ax.scatter(xy[:, 0], xy[:, 1], c=np.reshape(np.array(col), (1, -1)),
                           edgecolors=np.reshape(np.array(colB), (1, -1)), s=size, marker=marker,
                           label=label + str(k))
                else:
                    ax.scatter(xy[:, 0], xy[:, 1], c=np.reshape(np.array(col), (1, -1)),
                               edgecolors=np.reshape(np.array(colB), (1, -1)), s=size, marker=marker, label='')

    def sankeyPlot(self, paired):
        from pySankey import sankey
        import pandas as pd
        df = pd.DataFrame(data=paired, columns=['Old Cluster', 'New Cluster'])
        sankey.sankey(df['Old Cluster'], df['New Cluster'], 'Cluster Evolution through Movement of Skeleton Points', fontsize=15, aspect=2, figure_name=self.folder + 'Sankey Diagram of Cluster Evolution')



        # from matplotlib.sankey import Sankey
        #
        # if num == 1:
        #     transitions = [transitions]
        #
        # previous = np.ones((1, np.size(transitions, axis=1)))
        # for i,t in enumerate(transitions):
        #
        #     ranking = np.argsort(np.sum(t, axis=1))[::-1][:topClusters]
        #     for j,r in enumerate(ranking):
        #         fig = plt.figure()
        #         ax = fig.add_subplot(111, xticks=[], yticks=[])
        #         ax.set_title('Time Step ' + str(i) + ' - Cluster ' + str(r), fontsize=24)
        #         sankey = Sankey(ax=ax)
        #         total = sum(t[r, :])
        #         flows = []
        #         labels = []
        #         for k, f in enumerate(previous[:, j + 1]):
        #             if i == 0:
        #                 flows.append(f)
        #             else:
        #                 flows.append(f / total)
        #             labels.append('Old Cluster ' + str(k + 1))
        #
        #         for k, f in enumerate(t[r, :]):
        #             flows.append(- f / total)
        #             if k == 0:
        #                 labels.append('Unpaired')
        #             else:
        #                 labels.append('New Cluster ' + str(k))
        #
        #         sankey.add(flows=flows, orientations=[0 for _ in flows], labels=labels)
        #         sankey.finish()