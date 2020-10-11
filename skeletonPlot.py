import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import imageio


# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is in charge of plotting, either by means of a frozen 3D figure or by means of a GIF file,
# the constellation of skeleton points interior to each two-dimensional cluster along with the boundaries of each cluster.
# Álvaro Tomás Gil - UIUC 2020

class skeletonPlot:
    """This class is in charge of plotting, either by means of a frozen 3D figure or by means of a GIF file,
    the constellation of skeleton points interior to each two-dimensional cluster along with the boundaries of each cluster.
    For the inputs:
    bound3d: 2D array of 3D positions corresponding to the curves defining each cluster's boundaries for each of the
    parallel planes of constant Z in the domain
    skel: 2D array of 3D positions of skeleton points. In case skel2 is not None, this could also represent the positions
    of the points defining the trajectory between skeleton points with which the connectivity between them is examined.
    skel2: 2D array of 3D positions of skeleton points, in the case in which the connectivity between them is displayed
    labels: list of the same length as skel2, assigning a cluster label to each skeleton point
    folder: folder in which to save resulting plots"""
    def __init__(self, bound3d, skel, skel2=None, labels=None, folder=''):
        self.bound3d = bound3d
        self.skel = skel[np.argsort(skel[:, 2]), :][::-1]
        self.zs = np.unique(bound3d[:, 2])
        self.skel2 = skel2
        self.labels = labels
        self.folder = folder

        # Combinations of marker and marker boundary colors are made in order to increase the possibilities of marker types
        self.unique_labels = np.unique(self.labels)
        strength = np.linspace(0, 0.8, len(self.unique_labels))
        np.random.shuffle(strength)
        self.colors = [plt.cm.nipy_spectral(each) for each in strength]
        np.random.shuffle(strength)
        self.colorsB = [plt.cm.nipy_spectral(each) for each in strength]

        # Trajectory and boundary points are colored in terms of the Z value they appear in
        self.cm = cm.get_cmap('winter')
        normalized = (self.skel[:, 2] - np.min(self.zs)) / (np.ptp(self.zs))
        self.skelC = self.cm(normalized)
        normalized = (self.bound3d[:, 2] - np.min(self.zs)) / (np.ptp(self.zs))
        self.bounC = self.cm(normalized)

    def messyPlot(self, title='Topological Skelletonization of Clusters', labelB='BPs', labelS='Skeleton'):
        """This method defines a single 3D figure in which both cluster boundaries and skeleton points are displayed"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.set_size_inches(18.5, 9.5)
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.scatter(self.bound3d[:, 0], self.bound3d[:, 1], self.bound3d[:, 2], alpha=0.5, s=0.5, label=labelB)
        ax.scatter(self.skel[:, 0], self.skel[:, 1], self.skel[:, 2], c='red', s=0.5, alpha=1, label=labelS)
        ax.legend()

        X = self.bound3d[:, 0]
        Y = self.bound3d[:, 1]
        Z = self.bound3d[:, 2]
        # From https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

    def snapPlot(self, title, this=True, labelB='BPs', labelS='Skeleton'):
        """This method plots each two-dimensional domain separately, but joins all of them in a GIF animation which allows
        a 3D evolution of them to be visualized."""
        self.this = this
        self.labelB = labelB
        self.labelS = labelS

        def update(choose):
            fig, ax = plt.subplots()
            fig.set_size_inches(18.5, 9.5)

            self.plotInstance(choose, ax)
            ax.set_title('Z = ' + '{0:03f}'.format(choose), fontsize=24)
            ax.set_xlabel('x [m]', fontsize=18)
            ax.set_ylabel('y [m]', fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=15)
            plt.axis('equal')
            ax.set_xlim(np.min(self.bound3d[:, 0]), np.max(self.bound3d[:, 0]))
            ax.set_ylim(np.min(self.bound3d[:, 1]), np.max(self.bound3d[:, 1]))

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

        take = self.bound3d[:, 2] == choose
        if self.this:
            takeS = ((self.skel[:, 2] >= choose - dz) & (self.skel[:, 2] <= choose + dz))
        else:
            takeS = self.skel[:, 2] >= choose
        ax.scatter(self.bound3d[take, 0], self.bound3d[take, 1], c=self.bounC[take], s=1, label=self.labelB)
        ax.scatter(self.skel[takeS, 0], self.skel[takeS, 1], c=self.skelC[takeS], s=5, label=self.labelS, marker='D')

        if self.labels is not None:
            takeS2 = ((self.skel2[:, 2] >= choose - dz) & (self.skel2[:, 2] <= choose + dz))
            data = self.skel2[takeS2, :]
            labels = [self.labels[i] for i in range(len(self.labels)) if takeS2[i]]
            self.plotLabels(ax, data, labels)

    def plotLabels(self, ax, data, labels):

        for k, col, colB in zip(self.unique_labels, self.colors, self.colorsB):
            size = 15
            if k == 0:
                # Black used for noise.
                col = [1, 0, 0]
                size = 1

            class_member_mask = (labels == k)
            xy = data[class_member_mask]
            if len(xy) > 0:
                ax.scatter(xy[:, 0], xy[:, 1], c=np.reshape(np.array(col), (1, -1)),
                           edgecolors=np.reshape(np.array(colB), (1, -1)), s=50, marker='P',
                           label='SP of Cluster ' + str(k))
        ax.legend()
