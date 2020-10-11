import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import imageio


# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is in charge of plotting by means of a GIF file, the constellation of skeleton points
# interior to each two-dimensional cluster along with the Voronoi cell centers classified as cluster centers. On the one
# hand, skeleton positions are labeled according to their three-dimensional cluster, as well as according to whether their
# closest Voronoi cell is that of a cluster cell or not. On the other hand, Voronoi cell centers are labeled according to
# the cluster labeled assigned to them.
# Álvaro Tomás Gil - UIUC 2020

class voronoiPlot:
    """This class is in charge of plotting by means of a GIF file, the constellation of skeleton points
    interior to each two-dimensional cluster along with the Voronoi cell centers classified as cluster centers. On the one
    hand, skeleton positions are labeled according to their three-dimensional cluster, as well as according to whether their
    closest Voronoi cell is that of a cluster cell or not. On the other hand, Voronoi cell centers are labeled according to
    the cluster labeled assigned to them.
    For the inputs:
    data: 2D array of 3D positions corresponding to Voronoi cluster cell centers
    skel: 2D array of 3D positions of skeleton points.
    skelLabels: list of the same length as skel, assigning a cluster label to each skeleton point
    skelProx: list of the same length as skel, assigning a 1 to skeleton points whose closest Voronoi cell center is that
    of a cluster cell
    vorLabels: list of the same length as data, assigning a cluster label to each Voronoi cell center
    folder: folder in which to save resulting plots"""
    def __init__(self, data, skel, skelLabels, skelProx, vorLabels, folder=''):
        self.skel = skel
        self.skelLabels = skelLabels
        self.skelProx = skelProx
        self.vorLabels = vorLabels
        self.folder = folder

        self.zs = np.unique(skel[:, 2])

        #Project Voronoi cell centers onto the finite number of levels of Z
        self.data = np.array([0, 0, 0])
        self.vorLabels = []
        dz = abs(self.zs[1] - self.zs[0]) / 2

        for i, z in enumerate(self.zs):
            take = ((data[:, 2] < z + dz) & (data[:, 2] > z - dz))
            this = data[take, :2]
            this = np.hstack((this, z * np.ones((sum(take), 1))))
            self.data = np.vstack((self.data, this))

            newLabels = [vorLabels[i] for i in range(len(data)) if take[i]]
            self.vorLabels = self.vorLabels + newLabels
        self.data = self.data[1:, :]

        # Combinations of marker and marker boundary colors are made in order to increase the possibilities of marker types
        self.skelUniqueLabels = np.unique(self.skelLabels)
        strength = np.linspace(0, 0.8, len(self.skelUniqueLabels))
        np.random.shuffle(strength)
        self.skelColors = [plt.cm.autumn(each) for each in strength]
        np.random.shuffle(strength)
        self.skelColorsB = [plt.cm.autumn(each) for each in strength]

        #Marker colors for Voronoi cell centers
        self.vorUniqueLabels = np.unique(self.vorLabels)
        strength = np.linspace(0, 1, len(self.vorUniqueLabels))
        np.random.shuffle(strength)
        self.vorColors = [plt.cm.winter(each) for each in strength]

    def snapPlot(self, title='Topological Coincidence Between Skeleton Points and Voronoi Cell Centers'):
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
            ax.set_xlim(np.min(self.data[:, 0]), np.max(self.data[:, 0]))
            ax.set_ylim(np.min(self.data[:, 1]), np.max(self.data[:, 1]))

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

        take = self.data[:, 2] == choose
        takeS = ((self.skel[:, 2] >= choose - dz) & (self.skel[:, 2] <= choose + dz))

        skelCluster = [takeS[i] and self.skelProx[i] == 1 for i in range(len(takeS))]
        skelVoid = [takeS[i] and self.skelProx[i] == 0 for i in range(len(takeS))]
        data = self.data[take, :]
        skelClusterLabels = [self.skelLabels[i] for i in range(len(self.skelLabels)) if skelCluster[i]]
        skelVoidLabels = [self.skelLabels[i] for i in range(len(self.skelLabels)) if skelVoid[i]]
        vorLabels = [self.vorLabels[i] for i in range(len(self.vorLabels)) if take[i]]

        self.plotLabels(ax, data, vorLabels, self.vorColors, self.vorColors, None, size=15)
        self.plotLabels(ax, self.skel[skelCluster, :], skelClusterLabels, self.skelColors, self.skelColorsB, 'Coinciding Skeleton Points of Cluster ', marker='^')
        self.plotLabels(ax, self.skel[skelVoid, :], skelVoidLabels, self.skelColors, self.skelColorsB, 'Non-Coinciding Skeleton Points of Cluster ', marker='v')

    @staticmethod
    def plotLabels(ax, data, labels, colors, colorsB, label, marker='.', size=100):
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
