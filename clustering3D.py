import numpy as np
import matplotlib.pyplot as plt
from OPTICS import exploreOPTICS, reachabilityConvergence, compareWithRandom
from DBSCAN import exploreDBSCAN
from clusterPlot import clusterPlot
import time
import csv

# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is to determine the adequate pair or parameters epsilon and MinPts based on which a DBSCAN
# clustering analysis of several two-dimensional snapshots in Z is carried out. This script is also capable of evaluating
# the time consumed during its execution.
# Álvaro Tomás Gil - UIUC 2020


class clustering3D:
    """This is the main class of the script, which loads the dataset from its corresponding file, extracts the specified
    number of snapshots in Z to analyze, determines the optimum pair of MinPts, epsilon values based on a single snapshot,
    and sweeps throughout all of these planes in Z to carry out a DBSCAN clustering analysis. It also has the capability
    of plotting the results and evaluating its time consumption. For the inputs:
    filename: String which specifies file from which to specify the number of snapshots in Z to analyze
    take: Integer which specifies the number of snapshots in Z to analyze, starting from the midplane of the square duct.
    folder: folder in which to save resulting plots"""
    def __init__(self, filename, take=1, folder=''):
        self.folder = folder
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            self.uncut = np.array(list(reader)).astype(float)

        self.originalZ = np.load(folder + 'originalZ.npy')

        self.allZ = np.unique(self.uncut[:, 2])
        self.middle = int(np.floor(len(self.allZ) / 2))

        if take & 0x1 == 1:
            add = [a for a in range(int((take - 1) / 2) + 1)]
            subs = [-a for a in range(int((take - 1) / 2) + 1)]
        else:
            add = [a for a in range(int(take / 2) + 1)]
            subs = [-a for a in range(int(take / 2))]

        ind = np.unique(add + subs) + self.middle
        self.zs = np.ndarray.flatten(np.array([self.allZ[x] for x in ind]))
        self.data = np.array([0, 0, 0])
        self.original = np.array([0, 0, 0])

        for z in self.zs:
            take = self.uncut[:, 2] == z
            self.data = np.vstack((self.data, self.uncut[take, :]))
            self.original = np.vstack((self.original, self.originalZ[take, :]))
        self.data = self.data[1:, :]
        self.original = self.original[1:, :]
        np.save(folder + 'originalZ_cut', self.original)

        #Trial is the snapshot in Z which is chosen to obtain the optimum pair of MinPts and eps for the DBSCAN analysis
        self.trial = self.uncut[self.uncut[:, 2] == self.allZ[self.middle], :2]

    def MinPts(self, start=10, end=500, itera=30, xi0=0.05, known=0):
        """This method is in charge of determining the optimum value of MinPts to later be employed in the DBSCAN analysis
         of the dataset. In order to do so, it iterates throughout several values of MinPts. For the input:
         start: Integer representing the first value of MinPts to examine
         end: Integer representing the last value of MinPts to examine
         itera: Integer representing the number of values of MinPts to examine
         xi0: Float describing the value of xi to be employed in the OPTICS analysis of every examination
         known: Integer describing the known optimum MinPts in order to override the execution of this step."""
        startT = time.time()
        print('Determination of MinPts*')
        if known > 0:
            self.MinPts = known
            self.MinPtsT = 1569.539
        else:
            MinPts0 = np.linspace(start, end, itera)
            rC = reachabilityConvergence(filename=None, MinPts=MinPts0, xi=xi0, data=self.trial)
            rC.convergeConcaves(plot=False)
            self.MinPts = rC.convMinPts
            self.MinPtsT = time.time() - startT
        print('Execution Time: ' + str(round(self.MinPtsT, 3)))

    def meanReach(self, known=0):
        """This method is in charge of determining the optimum value of epsilon to later be employed in the DBSCAN analysis
        of the dataset. In order to do so, it calculates the mean reachability distance of a set of randomly distributed
        particles. For the input:
        known: Float describing the known optimum epsilon in order to override the execution of this step."""
        if known > 0:
            self.eps = known
            self.epsT = 106.725
        else:
            start = time.time()
            print('Determination of Mean Reachability of Random Distribution')
            cR = compareWithRandom(filename=None, MinPts=self.MinPts, xi=0.05, data=self.trial)
            cR.run()
            self.eps = cR.meanReach
            self.epsT = time.time() - start
        print('Execution Time: ' + str(round(self.epsT, 3)))

    def sweep(self):
        """Once both optimum parameters are known, this method performs the DBSCAN analysis of each of the snapshots in
        Z in question."""
        start = time.time()
        print('Sweep Along Z of Clustering Routine')

        self.labels = np.array([0])
        for z in self.zs:
            print('Z = ' + str(round(z, 3)))
            take = self.data[self.data[:, 2] == z, :]
            db = exploreDBSCAN(self.eps, self.MinPts, data=take)
            db.sweep()
            db.onlyCoreElements()
            self.labels = np.hstack((self.labels, db.labelsN[-1]))

        self.labels = self.labels[1:]
        self.sweepT = time.time() - start
        print('Execution Time: ' + str(round(self.sweepT, 3)))

    def visualize(self):
        """This method plots the clustering analysis' results."""
        cP = clusterPlot(self.data, self.labels, self.folder)
        cP.plotAll('3D DBSCAN Analysis - Z in ' + str(self.zs))

    def timeEvaluation(self):
        """This method analyzes the execution time associated to each of the steps in the analysis."""
        self.totalT = self.MinPtsT + self.epsT + self.sweepT

        labels = 'MinPts', 'meanReach', 'Sweep along Z'
        sizes = [self.MinPtsT, self.epsT, self.sweepT]
        explode = (0, 0, 0.1)

        fig1, ax1 = plt.subplots()
        wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True,
                                           startangle=180, rotatelabels=True)
        ax1.axis('equal')
        ax1.legend(wedges, labels, loc="best")
        plt.title('Execution Time Evaluation - ' + str(len(self.zs)) + ' Slices Taken - Total Time [m] = ' + str(
            round(self.totalT / 60, 3)))

