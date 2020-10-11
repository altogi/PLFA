import numpy as np
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from clusterPlot import clusterPlot
import csv

# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is designed to analyze a given dataset of particles with OPTICS in order to obtain the set
# of parameters eps and MinPts to be employed in a clustering scheme with DBSCAN. The script includes a simple class exploreOptics
# which performs the OPTICS analysis of the dataset for several values of MinPts and xi/eps, another class reachabilityConvergence
# which examines the evolution of the data's reachability plot with MinPts and obtains a value for MinPts at which the shape of the
# plot stabilizes, and another class compareWithRandom, which compares the reachability plot corresponding to such optimum MinPts
# with the mean reachability value resulting from applying an OPTICS analysis to a randomly distributed set of particles of the same
# length as the original dataset.
# Álvaro Tomás Gil - UIUC 2020

class exploreOPTICS:
    """This class is in charge of performing a preliminary OPTICS analysis of a 2D computational domain displaying particle positions. It
    has the capability of applying the clustering algorithm to the database for multiple different values of xi and MinPts, in order to
    analyze the effect of these parameters on the reachability plot of the data. For the input:
    xi: Float or list of floats, defining the xi or eps values based on which to perform xi or eps-clustering with OPTICS
    MinPts: Integer or list of integers, defining the list of MinPts based on which to perform xi or eps-clustering with OPTICS
    plots: Number of plots defining, for different values of xi or eps, the clustering results
    filename: In case data is None, file name from which to load data
    data: Data array on which to perform OPTICS analysis
    method: String defining which method to apply for OPTICS clustering
    distance: String defining which method to apply to compute distances within the algorithm"""
    def __init__(self, xi, MinPts, plots, filename=None, data=None, method='xi', distance='euclidean'):

        self.clust = []
        self.reach = []
        self.space = []
        self.dR = []
        self.normedR = []
        self.labels = []

        if np.isscalar(xi):
            self.xi = np.array([xi])
        else:
            self.xi = xi

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

        self.N = len(self.data)
        self.method = method
        self.distance = distance
        print('OPTICS: ' + str(self.N) + ' particles in data set.')
        self.n = np.array([0, 0, 0, 0])
        self.pl = [int(v) for v in np.linspace(0, len(self.xi) - 1, plots)]

        plt.figure()
        plt.scatter(self.data[:, 0], self.data[:, 1], s=0.8)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        if filename == None:
            plt.title('Particle Distribution' + ' - ' + str(self.N) + ' particles')
        else:
            plt.title(filename + ' - ' + str(self.N) + ' particles')

    def sweep(self):
        """For every combination of the specified values of xi/eps and MinPts, this method carries out the clustering analysis of the
        dataset based on those parameters. The results of such clustering are stored in clust, and the reachability distances of each
        clustering analysis are also calculated via reachability."""
        for j, u in enumerate(self.MinPts):
            for i, v in enumerate(self.xi):
                print('	' + str(i + 1) + 'th xi: ' + str(round(v, 4)) + '; ' + str(j + 1) + 'th MinPts: ' + str(
                    np.floor(u)))
                if self.method == 'xi':
                    c = OPTICS(min_samples=int(u), xi=v, algorithm='auto', metric=self.distance)
                else:
                    c = OPTICS(min_samples=int(u), eps=v, algorithm='auto', metric=self.distance)
                c.fit(self.data)

                # While as clust is a list of OPTICS objects, labels contains the labels of such objects
                self.clust.append(c)
                self.labels.append(c.labels_)

                # Every row of n contains: [xi, MinPts, Number of Clusters, Percentage of Noise]
                nC = [v, u, len(np.unique(np.array(c.labels_))) - 1, len(np.argwhere(c.labels_ == -1)) * 100 / self.N]
                self.n = np.vstack((self.n, nC))

                if i in self.pl:
                    # Plot the clustering disposition of the radii selected in array pl
                    title = 'OPTICS with MinPts = ' + str(np.floor(u)) + ' and xi = ' + str(round(v, 4)) + '. ' + str(
                        nC[-2]) + ' clusters detected.'
                    cP = clusterPlot(self.data, c.labels_)
                    cP.plotAll(title)
            # cP.sizeHistogram(title)

            self.reachability(u)
        self.n = self.n[1:, :]

    def reachability(self, pts):
        """This function computes and stores the reachability distances for clustering analyses associated with the
        MinPts specified via pts. Note that the reachability plot will not depend on xi nor eps."""
        ofPts = np.ndarray.flatten(np.argwhere(self.n[1:, 1] == pts))
        c = self.clust[ofPts[-1]]
        reachability = c.reachability_[c.ordering_]
        space = np.arange(len(self.data))
        minR = np.nanmin(reachability[np.isfinite(reachability)])

        self.reach.append(reachability)
        self.space.append(space)
        self.dR.append(np.diff(reachability) / np.diff(space))
        self.normedR.append(reachability / minR)

    def plot(self, forXi=True, forMinPts=True):
        """This method is in charge of plotting the evolution of the number of detected clusters and the percentage of
        particles labelled as noise with a varying xi/eps and/or a varying MinPts"""
        if forXi:
            for i, v in enumerate(self.MinPts):
                fig, ax1 = plt.subplots()
                color = 'tab:red'
                ax1.set_xlabel('xi [-]')
                ax1.set_ylabel('Clusters Detected [-]', color=color)
                ax1.plot(self.xi, self.n[self.n[:, 1] == v, 2], color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_title('OPTICS with MinPts = ' + str(v) + ' and Euclidean Distance')

                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.set_ylabel('Percentage of Noise [%]', color=color)
                ax2.plot(self.xi, self.n[self.n[:, 1] == v, 3], color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                fig.tight_layout()

        if forMinPts:
            for i, v in enumerate(self.xi):
                fig, ax1 = plt.subplots()
                color = 'tab:red'
                ax1.set_xlabel('MinPts [-]')
                ax1.set_ylabel('Clusters Detected [-]', color=color)
                ax1.plot(self.MinPts, self.n[self.n[:, 0] == v, 2], color=color)
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_title('OPTICS with xi = ' + str(v) + ' and Euclidean Distance')

                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.set_ylabel('Percentage of Noise [%]', color=color)
                ax2.plot(self.MinPts, self.n[self.n[:, 0] == v, 3], color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                fig.tight_layout()

        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(self.n[:,0], self.n[:,1], self.n[:,2], marker = 'o', c = self.n[:,2], cmap = 'inferno')
        # ax.set_title('Clusters Detected - OPTICS with Euclidean Distance')
        # ax.set_xlabel('xi [-]')
        # ax.set_ylabel('MinPts [-]')

    def PlotReachabilityWithLabels(self):
        """This method plots the reachability distances associated to all possible values of MinPts, also including
         within the plot how the cluster label assignment varies with xi/eps."""
        for pts in self.MinPts:

            plt.figure(figsize=(10, 7))
            plt.ylabel('Reachability')
            plt.title('Reachability Plot - MinPts: ' + str(pts))

            ofPts = np.ndarray.flatten(np.argwhere(self.n[:, 1] == pts))
            space = self.space[ofPts[0]]
            reachability = self.reach[ofPts[0]]
            minR = np.nanmin(reachability[np.isfinite(reachability)])
            maxR = np.nanmax(reachability[np.isfinite(reachability)])
            lines = np.linspace(minR + 0.1 * (maxR - minR), maxR - 0.1 * (maxR - minR), len(ofPts))
            plt.plot(space, reachability)

            for i in range(len(lines)):

                if len(lines) == 1:
                    dLine = 0.05 * (maxR - minR)
                else:
                    dLine = 0.1 * (lines[1] - lines[0])
                c = self.clust[ofPts[i]]
                y = lines[i] * np.ones((len(space), 1))
                labels = c.labels_[c.ordering_]
                unique_labels = np.unique(np.array(labels))
                colors = [plt.cm.viridis(each) for each in np.linspace(0, 1, len(unique_labels))]

                xi = self.n[ofPts[i], 0]
                font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
                plt.text(space[0], lines[i] + dLine, 'xi = ' + str(round(xi, 5)), fontdict=font)

                for k, col in zip(unique_labels, colors):
                    size = 1.6
                    if k == -1:
                        # Black used for noise.
                        col = [128 / 256, 128 / 256, 128 / 256]
                        size = 1

                    class_member_mask = (labels == k)
                    plt.plot(space[class_member_mask], y[class_member_mask], 'o', markerfacecolor=tuple(col),
                             markersize=size, markeredgecolor=tuple(col))


class reachabilityConvergence:
    """This class is employed to examine possible ways of convergence of the reachability plot resulting from several OPTICS
    based on different values of MinPts. All the calculations are performed within class exploreOPTICS, this class
    is only in charge of processing and displaying the results. For the input:
    filename: In case data is None, file name from which to load data
    data: Data array on which to perform OPTICS analysis
    xi: Float or list of floats, defining the xi or eps values based on which to perform xi or eps-clustering with OPTICS
    MinPts: Integer or list of integers, defining the list of MinPts based on which to perform xi or eps-clustering with OPTICS"""
    def __init__(self, filename=None, MinPts=100, xi=0.05, data=None):
        self.MinPts = MinPts
        self.Optics = exploreOPTICS(xi, self.MinPts, 0, filename, data)
        self.errR = []
        self.Optics.sweep()

    def converge(self):
        """This class is in charge of plotting the reachability plots for different values of MinPts in order to examine
        their evolution in shape. Note however that the reachability distances plotted for each value of MinPts correspond
        to the reachability distances normed by the minimum reachability distance for that value of MinPts."""
        plt.figure(figsize=(10, 7))
        plt.ylabel('Reachability')
        plt.title('Normed Reachability Plot')
        colors = [plt.cm.inferno(each) for each in np.linspace(0, 1, len(self.MinPts))]
        for i, u in enumerate(self.MinPts):
            legend = 'MinPts = ' + str(np.floor(u))
            space = self.Optics.space[i]
            plt.plot(space, self.Optics.normedR[i], label=legend, color=tuple(colors[i]))

            if i > 0:
                v1 = np.nan_to_num(self.Optics.normedR[i])
                v0 = np.nan_to_num(self.Optics.normedR[i - 1])
                self.errR.append(np.linalg.norm(v1 - v0))
        plt.legend(loc='upper left')

        #This secondary plot displays how the normed reachability varies from MinPts to MinPts, taking into account the
        #dataset as a whole.
        plt.figure()
        plt.title('Variation in Normed Reachability w.r.t. Previous MinPts')
        plt.ylabel('Variation in Normed Reachability')
        plt.xlabel('MinPts')
        plt.plot(self.MinPts[1:], self.errR)

    def convergeDerivative(self):
        """This method is in charge of examining how the derivative of the reachability varies with MinPts."""
        plt.figure(figsize=(10, 7))
        plt.title('Derivative of Reachability')
        for i, u in enumerate(self.Optics.dR):
            legend = 'MinPts = ' + str(np.floor(self.Optics.MinPts[i]))
            space = self.Optics.space[0]

            plt.plot(space[1:], u, label=legend)
        plt.legend(loc='upper left')

    def convergeConcaves(self, plot=True):
        """This methods analyzes how the number of dents within the reachability plot varies with increasing value of
        MinPts. In order to determine what consitutes a dent, the function employs the gradient of the reachability.
        Moreover, this method determines at which MinPts the number of dents within the reachability plot more or less
        stabilizes."""
        self.concaves = []

        for i, v in enumerate(self.Optics.reach):
            diffR = np.gradient(v)
            conc = 0
            hasDown = False

            for j, u in enumerate(diffR):
                if u < 0 and not hasDown:
                    hasDown = True

                if u > 0 and hasDown:
                    conc += 1
                    hasDown = False

            self.concaves.append(conc)

        diffConv = np.gradient(self.concaves)
        diffMinPts = np.gradient(self.MinPts)
        converged = np.ndarray.flatten(np.argwhere(np.abs(diffConv / diffMinPts) < 0.1))
        convMinPts = [self.MinPts[converged[0]], self.concaves[converged[0]]]
        self.convMinPts = int(convMinPts[0])

        if plot:
            plt.figure(figsize=(10, 7))
            plt.title('Number of Concave-Up Occurences in Reachability Plot - N = ' + str(self.Optics.N))
            plt.plot(self.MinPts, self.concaves)
            plt.xlabel('MinPts [-]')
            plt.ylabel('Concave-Up Occurences [-]')
            plt.plot(convMinPts[0], convMinPts[1], 'o', color='red')
            font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
            plt.text(convMinPts[0], convMinPts[1] + 5, 'MinPts* = ' + str(int(convMinPts[0])), fontdict=font)


class compareWithRandom:
    """This class is in charge of, given a single MinPts value, compare the reachability plot resulting from employing
    it in an OPTICS analysis with the reachability plot that results from applying OPTICS to a randomly distributed set
    of particles of the same lenght as the original dataset. For the input:
    filename: In case data is None, file name from which to load data
    data: Data array on which to perform OPTICS analysis
    xi: Float or list of floats, defining the xi or eps values based on which to perform xi or eps-clustering with OPTICS
    MinPts: Integer or list of integers, defining the list of MinPts based on which to perform xi or eps-clustering with OPTICS
    plots: Boolean determing whether to plot the reachability plots of the dataset and of the randomly distributed set of
    particles."""
    def __init__(self, filename=None, MinPts=100, xi=0.05, data=None, plots=False):
        # At this point it is assumed that MinPts is a single value
        self.MinPts = MinPts
        if np.isscalar(xi):
            self.xi = np.array([xi])
        else:
            self.xi = xi
        self.Optics = exploreOPTICS(xi, MinPts, 0, filename, data)
        self.Optics.sweep()
        self.space = self.Optics.space[0]
        self.reach = self.Optics.reach[0]
        self.N = self.Optics.N
        self.plots = plots

    def randomlyDistributed(self):
        """This method generates a random distribution of particles within the same domain as the original dataset and
        executes an OPTICS analysis on it."""
        maxs = np.array([np.nanmax(self.Optics.data[:, 0]), np.nanmax(self.Optics.data[:, 1])])
        mins = np.array([np.nanmin(self.Optics.data[:, 0]), np.nanmin(self.Optics.data[:, 1])])
        self.points = np.multiply(maxs - mins, np.random.rand(self.N, 2)) + mins
        self.OpticsR = exploreOPTICS(self.xi[0], self.MinPts, 0, filename='Random Distribution', data=self.points)
        self.OpticsR.sweep()
        self.meanReach = np.mean(self.OpticsR.reach[0][np.isfinite(self.OpticsR.reach[0])])

    def compareReachabilities(self):
        """This method plots and compares the reachability distances of the original dataset and those of the randomly
        distributed set of particles."""
        plt.figure(figsize=(10, 7))
        plt.ylabel('Reachability')
        plt.title('Comparison between Reachability Plots - N = ' + str(self.N))
        plt.plot(self.Optics.space[0], self.Optics.reach[0], label='Preferential Distribution')
        plt.plot(self.OpticsR.space[0], self.OpticsR.reach[0], label='Random Distribution')
        plt.legend(loc='upper left')

    def determineXifromRandom(self):
        """Based on the mean reachability obtained from randomly distributing the same number of particles in the same
        domain area, this method determines which value of xi mostly classifies particles with reachability higher than
        such mean value as noise"""

        self.xiScore = np.zeros(len(self.xi))
        for i, u in enumerate(self.xi):
            c = self.Optics.clust[i]
            labels = c.labels_[c.ordering_]
            for j, v in enumerate(labels):
                if v == -1 and self.isNoise[j] == -1:
                    self.xiScore[i] += 1

                if v != -1 and self.isNoise[j] == 1:
                    self.xiScore[i] += 1

        self.xiScore = self.xiScore / (self.N * 0.01)
        self.optXi = self.xi[np.argmax(self.xiScore)]

        self.Optics.PlotReachabilityWithLabels()
        plt.plot(self.space, self.meanReach * np.ones(len(self.space)))

        plt.figure()
        plt.title('Affinity Score for each Xi')
        plt.ylabel('Score [%]')
        plt.xlabel('xi [-]')
        plt.plot(self.xi, self.xiScore)

    def plotOptimumXi(self):
        """This method plots the reachability plot of the employed dataset and also displays how the optimum xi value
        obtained in the previous method groups the dataset into different clusters."""
        plt.figure(figsize=(10, 7))
        plt.ylabel('Reachability')
        plt.title('Reachability Plot - MinPts =  ' + str(int(self.MinPts)) + ' & xi* = ' + str(self.optXi))

        plt.plot(self.space, self.reach)

        ofXi = np.ndarray.flatten(np.argwhere(self.xi == self.optXi))[0]
        c = self.Optics.clust[ofXi]
        labels = c.labels_[c.ordering_]
        unique_labels = np.unique(np.array(labels))
        colors = [plt.cm.Dark2(each) for each in np.linspace(0, 1, len(unique_labels))]
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        plt.text(self.space[5], self.meanReach * 1.02, 'Mean Reachability of Random Distribution', fontdict=font)

        for k, col in zip(unique_labels, colors):
            size = 1.6
            if k == -1:
                # Black used for noise.
                col = [128 / 256, 128 / 256, 128 / 256]
                size = 1

            class_member_mask = (labels == k)
            plt.plot(self.space[class_member_mask], self.meanReach * np.ones(len(np.argwhere(class_member_mask))), 'o',
                     markerfacecolor=tuple(col), markersize=size, markeredgecolor=tuple(col))

    def easyLabel(self):
        """This method simply applies a label value of -1 (Noise) to particles with reachability greater than the obtained
        mean reachability and a label value of 1 (Cluster) to the rest."""
        self.isNoise = np.ones(len(self.space))
        for i, u in enumerate(self.space):
            if self.reach[i] > self.meanReach:
                self.isNoise[i] = -1
        # ORDERING OF LABELS IS NOT THE SAME AS THE ONE IN DATA!!!
        c = self.Optics.clust[0]
        self.easyData = self.Optics.data[c.ordering_, :]

    def run(self):
        """Execution of the previous methods"""
        self.randomlyDistributed()
        if self.plots:
            self.compareReachabilities()
        self.easyLabel()
        if len(self.xi) > 1:
            self.determineXifromRandom()
            self.plotOptimumXi()

