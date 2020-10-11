import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import csv
from clusterPlot import clusterPlot


# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is designed to execute a DBSCAN clustering analysis of a 2D snapshot in Z displaying
# particle positions, taking into account multiple combinations of eps and MinPts. This script is also capable of
# plotting the results of such clustering routine and excluding non-core elements from the cluster label.
# Álvaro Tomás Gil - UIUC 2020

class exploreDBSCAN:
    """This class is in charge of performing a  DBSCAN analysis of a 2D computational domain displaying particle positions. It
    has the capability of applying the clustering algorithm to the database for multiple different values of eps and MinPts,
    in order to analyze the effect of these parameters on the clustering of the data. For the input:
    eps: Float or list of floats, defining the eps values based on which to perform clustering with DBSCAN
    MinPts: Integer or list of integers, defining the list of MinPts based on which to perform clustering with DBSCAN
    plots: Number of plots defining, for different values of xi or eps, the clustering results
    filename: In case data is None, file name from which to load data
    data: Data array on which to perform OPTICS analysis"""
    def __init__(self, eps, MinPts, plots=0, filename=None, data=None):

        if np.isscalar(eps):
            self.eps = np.array([eps])
        else:
            self.eps = eps

        if np.isscalar(MinPts):
            self.MinPts = np.array([MinPts])
        else:
            self.MinPts = MinPts

        if data is not None:
            self.data = data
        else:
            with open(filename, 'r') as f:
                reader = csv.reader(f, delimiter=',')
                self.data = np.array(list(reader)).astype(float)

        self.clust = []
        self.labels = []
        self.n = np.array([0, 0, 0])
        self.N = len(self.data)
        self.pl = [int(v) for v in np.linspace(0, len(self.eps) - 1, plots)]
        print('DBSCAN: ' + str(self.N) + ' particles in data set.')

    def sweep(self):
        """For every combination of the specified values of eps and MinPts, this method carries out the clustering analysis of the
        dataset based on those parameters. The results of such clustering are stored in clust, and labels."""
        for i, v in enumerate(self.eps):
            for j, u in enumerate(self.MinPts):
                print('	' + str(i + 1) + 'th epsilon: ' + str(round(v, 4)) + '; ' + str(j + 1) + 'th MinPts: ' + str(
                    np.floor(u)))
                c = DBSCAN(eps=v, min_samples=np.floor(u), metric='minkowski', p=2)
                c.fit(self.data)
                self.clust.append(c)
                self.labels.append(c.labels_)

                #Every row of n includes [eps, MinPts, Number of Clusters]
                nC = [v, u, len(np.unique(np.array(c.labels_)))]
                self.n = np.vstack((self.n, nC))

                if i in self.pl:
                    # Plot the clustering disposition of the radii selected in array pl
                    title = 'DBSCAN with MinPts = ' + str(np.floor(u)) + ' and $\epsilon$ = ' + str(
                        round(v, 4)) + '. ' + str(nC[-1]) + ' clusters detected.'
                    cP = clusterPlot(self.data, c.labels_)
                    cP.plotAll(title)
        # cP.sizeHistogram(title)

    def onlyCoreElements(self):
        """This method defines a new list of labels in which non-core elements previously included within clusters are
        now excluded from any cluster and labelled as noise."""
        self.labelsN = self.labels
        for i, c in enumerate(self.clust):
            cores = c.core_sample_indices_
            take = np.setdiff1d(np.arange(0, self.N), cores)
            self.labelsN[i][take] = -1

    def plot(self):
        """This function plots the evolution of the detected number of particles with the neighborhood radius eps"""
        self.n = self.n[1:][:]
        fig = plt.figure()
        if len(self.MinPts) == 1:
            plt.plot(self.eps, self.n[:, 2])
            plt.title('DBSCAN with MinPts = ' + str(self.MinPts[0]) + ' and Euclidean Distance')
            plt.xlabel('$\epsilon$ [-]')
            plt.ylabel('Clusters Detected [-]')
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.n[:, 0], self.n[:, 1], self.n[:, 2], marker='o', c=self.n[:, 2], cmap='inferno')
            ax.set_title('Clusters Detected - DBSCAN with Euclidean Distance')
            ax.set_xlabel('$\epsilon$ [-]')
            ax.set_ylabel('MinPts [-]')
        plt.show()

    # def isolateCluster(self, label):
    #     isolated = np.array([0, 0])
    #     for c in self.clust:
    #         take = self.data[c.labels_ == label]
    #         isolated = np.vstack((isolated, take))
    #
    #     isolated = isolated[1:]
    #     return isolated
