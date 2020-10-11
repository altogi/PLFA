import numpy as np
from sklearn.neighbors import NearestNeighbors
from temporalPlot import temporalPlot
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is designed to analyze an already-processed pair of sets of particle positions, for adjacent instants of time.
# The carried out analysis is basically focused on comparing neighboring topologies for skeleton points of both time instants.
# For each skeleton point, the distance to the closest boundary particle in every direction is measured, and then each skeleton
# point of the first time instant is paired with the skeleton point of the second time frame which has the most similar set
# of measured distances and which is within an area of expected translation. The main output of this analysis is an array
# of transitions which, for every cluster label of the first time frame, takes into account the cluster labels which they
# theoretically have adopted in the second time frame.
# Álvaro Tomás Gil - UIUC 2020

class temporalTracker:
    """This class is designed to analyze an already-processed pair of sets of particle positions, for adjacent instants of time.
    The carried out analysis is basically focused on comparing neighboring topologies for skeleton points of both time instants.
    For each skeleton point, the distance to the closest boundary particle in every direction is measured, and then each skeleton
    point of the first time instant is paired with the skeleton point of the second time frame which has the most similar set
    of measured distances and which is within an area of expected translation. The main output of this analysis is an array
    of transitions which, for every cluster label of the first time frame, takes into account the cluster labels which they
    theoretically have adopted in the second time frame.
    For the inputs:
    pairs: List of two ints, containing the time instants to compare
    v: Float representing the channel flow bulk velocity associated to the datasets
    eta: Kolmogorov length scale, used as a reference length
    urms: Fluctuating particle velocity, proportional to the flow bulk velocity
    ksens: Float, which when multiplied with the distance that a particle at v travels between both time instants, defines
    the neighborhood radius with which to carry out the distance measurements for each skeleton point
    kfocus: Float, which when multiplied with the distance that a particle at v travels between both time instants, describes
    the area of interest in which to look for skeleton points in the second time frame which have similar distance measures
    dirs: Number of distance measurements to carry out for each skeleton point
    root: Address from which to retrieve time instant data"""
    def __init__(self, pair, v=7.7 * 0.15e-3 / 100, eta=70.59*10**-6, urms=0.1, ksens=63.75, kfocus=1, dirs=10, dt=0.15, root='./Data/v78/t_'):
        self.v = v
        self.pair = pair

        self.add = [root + str(p) + '/' for p in pair]
        self.d = v * (pair[1] - pair[0])
        self.sensorRadius = ksens * eta

        self.regionRadius = [urms * self.d, kfocus]

        self.sk = [np.load(a + 'skeletonize.npy') for a in self.add]
        self.bound = [np.load(a + 'bound3d.npy') for a in self.add]
        self.labels = [np.load(a + 'SKlabels.npy') for a in self.add]
        self.volumes = [np.load(a + 'DBSCANVolumes.npy') for a in self.add]
        self.nbrs = [NearestNeighbors(radius=self.sensorRadius, algorithm='auto').fit(b) for b in self.bound]
        self.zs = np.unique(self.sk[0][:, 2])
        self.connections = np.array([0, 0, 0])
        self.connAngles = []

        #Transitions has as many rows as labels in the old time instant, and as many columns as labels in the new time
        # instant. By taking the i-th row of the dataset, the j-th column represents how many skeleton points from the
        # cluster label i in the old time frame are paired with a cluster label j in the new time instant
        self.transitions = np.zeros((len(np.unique(self.labels[0])), len(np.unique(self.labels[1])) + 1))
        self.paired = np.hstack((np.reshape(self.labels[0], (-1, 1)), np.zeros((len(self.labels[0]), 1))))
        self.pairedString = [['t = ' + str(self.pair[0] * dt / 100) + ' - Cluster ' + str(int(i)), 0] for i in self.labels[0]]

        self.dirs = np.array([0, 0, 0])
        self.corresponding = np.array([])
        for theta in np.linspace(0, 2 * np.pi, dirs):
            x = np.cos(theta)
            y = np.sin(theta)
            z = 0
            self.dirs = np.vstack((self.dirs, np.array([x, y, z])))
            self.corresponding = np.append(self.corresponding, theta)
        self.dirs = self.dirs[1:, :]

    def run(self, plot=True, info=False):
        self.proximities = [self.proximities2D(self.sk[i], i) for i in range(2)]

        self.aveD = np.mean(self.proximities[0], axis=1)  # Mean distance to neighboring boundary particles
        self.regionRadii = self.regionRadius[0] + self.regionRadius[1] * self.aveD
        self.focus = NearestNeighbors(radius=np.max(self.regionRadii), algorithm='auto').fit(self.sk[1])

        for i, sk in enumerate(self.sk[0]):
            self.forSkelPoint(i)

        print('Out of ' + str(len(self.sk[0])) + ' skeleton points of the first time frame, ' + str(sum(self.transitions[:, 0])) + ' (' + str(
            round(sum(self.transitions[:, 0]) * 100 / len(self.sk[0]),
                  3)) + ' %) are not paired with another skeleton point in the second time frame')

        self.unpaired = sum(self.transitions[:, 0]) * 100 / len(self.sk[0])
        print('Connecting Vectors are on average oriented ' + str(round(np.mean(self.connAngles) * 180 / np.pi, 3)) + ' degrees wrt the X axis')

        if info:
            for l in np.unique(self.labels[0]):
                members = sum(self.labels[0] == l)
                topNew = np.argsort(self.transitions[l - 1, :])[::-1][:5]
                counts = np.sort(self.transitions[l - 1, :])[::-1][:5]
                print('For the ' + str(members) + ' skeleton points in the first time instant with label ' + str(l) + ': ')
                for i in range(5):
                    if counts[i] > 0:
                        if topNew[i] == 0:
                            add = ' ' + str(counts[i]) + ' ( ' + str(round(counts[i] * 100 / members, 3)) + ' %) are not paired with any skeleton point from the next time instant'
                        else:
                            add = ' ' + str(counts[i]) + ' ( ' + str(round(counts[i] * 100/ members, 3)) + ' %) are paired in the next time instant with cluster label ' + str(topNew[i])
                        print(add)

        if plot:
            self.plotResults()

    def proximities2D(self, skel, i):
        """This method measures, for every skeleton point in skel, the distances to the closest boundary particle in each
        direction. These distances in each direction are returned via proximities, which is an array witg as many rows
        as skel and as many columns as dirs. For the input:
        skel: 2D array of 3D skeleton positions
        i: Int dictating which time instant to take into account"""
        distAll, indiAll = self.nbrs[i].radius_neighbors(skel)
        proximities = np.zeros((1, len(self.dirs)))
        for j, x in enumerate(skel):
            dist = distAll[j]
            if len(dist) > 0:
                indi = indiAll[j]
                vecs = self.bound[i][indi, :] - x

                dots = np.matmul(vecs, np.transpose(self.dirs))  # As many rows as len(vecs), as many columns as len(dirs)
                maxDot = np.argmax(dots, axis=1)  # as many elements as vecs, indexes dirs
                covered = []  # indexes vecs, same len as dirs
                for k in range(len(self.dirs)):
                    intheway = np.ndarray.flatten(np.argwhere(maxDot == k))  # indexes vecs
                    if len(intheway) == 1:
                        covered.append(intheway[0])
                    elif len(intheway) > 1:
                        d = dist[intheway]
                        take = np.argmin(d)
                        covered.append(intheway[take])
                    else:
                        covered.append(-1)

                dist = np.append(dist, self.sensorRadius)
                distances = dist[covered]
            else:
                distances = self.sensorRadius * np.ones((1, len(self.dirs)))
            proximities = np.vstack((proximities, distances))
        return proximities[1:]


    def forSkelPoint(self, i, dt=0.15):
        """For the skeleton point of the first time frame specified by index i, this method looks for the neighboring
        skeleton points of the next time frame close to the points expected position, and extracts the skeleton point of
        the next time frame which has the closest measures of proximitiy distances, as well as its cluster label. dt is
        the actual time step in ms"""
        r = self.sk[0][i, :]
        label = self.labels[0][i]
        distances = self.proximities[0][i, :]
        rexpected = r + self.d * np.array([1, 0, 0])

        possible = self.focus.radius_neighbors(np.reshape(rexpected, (1, -1)), return_distance=False, radius=self.regionRadii[i])
        possible = [p for p in possible[0] if self.sk[1][p, 2] == r[2]]
        toCompare = self.proximities[1][possible, :]

        deviations = np.sum((toCompare - distances)**2, axis=1)
        if len(deviations) == 0:
            newLabel = 0
        else:
            closest = possible[np.argmin(deviations)]
            newLabel = self.labels[1][closest]
            connection = np.linspace(r, self.sk[1][closest, :], 100)
            self.connections = np.vstack((self.connections, connection))
            angle = np.arctan2(connection[-1, 1] - connection[0, 1], connection[-1, 0] - connection[0, 0])
            self.connAngles.append(angle)
        self.transitions[label - 1, newLabel] += 1
        self.paired[i, 1] = newLabel
        if newLabel == 0:
            self.pairedString[i][1] = 't = ' + str(self.pair[1] * dt / 100) + ' ms - Unpaired'
        else:
            self.pairedString[i][1] = 't = ' + str(self.pair[1] * dt / 100) + ' ms - Cluster ' + str(newLabel)

    def plotResults(self):
        self.plot = temporalPlot(self.sk, self.bound, self.labels, self.connections, self.add[1])
        self.plot.snapPlot()
        self.plot.sankeyPlot(self.pairedString)

class temporalTrackerGlobal:
    """This class simply invokes temporalTracker for more than one pair of time instants, and is able to track topology
    evolutions over a greater extent of time.
    For the inputs:
    pairs: 2D list where each element is a pair of time instants to examine
    dt: Time step for 100 iterations in ms
    t_Kolmogorov: Kolmogorov Time scale in ms
    eta: Kolmogorov Length scale in m"""
    def __init__(self, pairs, dt=0.15, eta=70.59*10**-6, t_Kolmogorov=1/3, root='./Data/v78/t_', plot=True):
        self.pairs = pairs
        self.trackers = [0 for _ in pairs]
        self.transitions = [0 for _ in pairs]
        self.nClusters = 0
        self.volumes = []
        self.times = []
        self.unpaired = [0 for _ in pairs]
        self.eta = eta
        self.tK = t_Kolmogorov
        for i, p in enumerate(pairs):
            tt = temporalTracker(p, root=root, dt=dt)
            tt.run(plot=plot)
            self.trackers[i] = tt
            self.transitions[i] = tt.transitions
            self.nClusters = max(self.nClusters, len(tt.volumes[0]))
            self.unpaired[i] = tt.unpaired
            self.volumes.append([t for t in tt.volumes[0]])
            self.times.append(p[0] * dt / 100)
            if i==len(pairs) - 1:
                self.volumes.append([t for t in tt.volumes[1]])
                self.times.append(p[1] * dt / 100)
                self.nClusters = max(self.nClusters, len(tt.volumes[1]))

    def trackVolumes(self, plot=True):
        """This particular method is in charge of visualizing the evolution of the volumes of the smaller clusters in the
        domain, by connecting them with other smaller clusters across time or labeling them as fully unpaired or as
        joining the largest cluster in the domain.
        For the inputs:
        plot: Boolean determining whether to plot or not"""

        self.lifetimes = np.zeros((1, len(self.times), 2))
        # Structure of lifetimes: each row corresponds to the lifetime of a single cluster, each column to a particular
        # time instant. In the third dimension, the first index corresponds to the volumes which the cluster adopts
        # in its lifetime, the second to an integer flag marking 1 when the cluster has dissociated, 2 when it has merged with
        # another small cluster or 3 if it has merged with the biggest cluster.
        taken = np.zeros((len(self.times), self.nClusters))
        # Structure of taken: In order to know whether a cluster being traced is merging with a cluster already being traced,
        # taken, having as many rows as time instants and as many columns as possible clusters, determines whether the
        # life of a cluster has been traced or not, by placing a 1 in those time instants traced for a particular cluster.

        for i,t in enumerate(self.times[:-1]):
            for j in range(np.size(self.transitions[i], 0)):
                life = np.zeros((1, len(self.times), 2))
                this = j
                if taken[i, this] == 0:
                    for k in range(i, len(self.times)):
                        if max(life[0, :, 0]) > 15*life[0, np.argmax(life[0, :, 0]) - 1, 0] and life[0, np.argmax(life[0, :, 0]) - 1, 0] > 0:
                            print('e')
                        bigCluster = np.argmax(self.volumes[k])
                        taken[k, this] = 1
                        if this != bigCluster:
                            life[0, k, 0] = self.volumes[k][this]
                            if k < len(self.times) - 1:
                                transitions = self.transitions[k]
                                next = np.argmax(transitions[this, :])
                                # nextBig = np.argmax(self.volumes[k + 1])
                                nextBig = np.argmax(np.sum(transitions[:, 1:], 0))
                                if next == 0:
                                    life[0, k, 1] = 1
                                    self.lifetimes = np.vstack((self.lifetimes, life))
                                    break
                                elif next == nextBig + 1:
                                    life[0, k, 1] = 3
                                    self.lifetimes = np.vstack((self.lifetimes, life))
                                    break
                                else:
                                    if taken[k + 1, next - 1] == 0:
                                        this = next - 1
                                    else:
                                        life[0, k + 1, 1] = 2
                                        life[0, k + 1, 0] = self.volumes[k + 1][next - 1]
                                        self.lifetimes = np.vstack((self.lifetimes, life))
                                        break
                            else:
                                self.lifetimes = np.vstack((self.lifetimes, life))
                                break
                        else:
                            break
        self.lifetimes = self.lifetimes[1:, :]

        times = [t / self.tK for t in self.times]
        self.age = np.zeros((np.size(self.lifetimes, 0), 2))
        self.aveV = np.zeros(np.size(self.lifetimes, 0))
        for i, l in enumerate(self.lifetimes[:, :, 0]):
            alive = np.flatnonzero(l)
            lifetime = [times[t] for t in alive]
            self.age[i, 0] = lifetime[-1] - lifetime[0]
            self.aveV[i] = np.mean(l[alive])

            if sum(self.lifetimes[i, :, 1]) > 0:
                alive = np.flatnonzero(self.lifetimes[i, :, 1])
                self.age[i, 1] = self.lifetimes[i, alive, 1]

        if plot:
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(16, 10)
            ax.set_xlabel(r'$\frac{t}{t_\eta}$ [-]', fontsize=24)
            ax.grid(True)
            ax.set_ylabel("\nCluster Volume " + r'$\frac{V}{\eta^3}$ [-]', fontsize=24)
            ax.tick_params(axis='both', labelsize=15)
            ax.set_title('Evolution of Volume of Smaller Clusters', fontsize=20)
            fig.tight_layout()

            cmap = cm.get_cmap('Dark2')
            for i, l in enumerate(self.lifetimes[:, :, 0]):
                alive = np.flatnonzero(l)
                color = cmap(np.random.random())
                lifetime = [times[t] for t in alive]
                volumes = l[alive] / self.eta ** 3
                ax.plot(lifetime, volumes, color=color, linewidth=2)

                if sum(self.lifetimes[i, :, 1]) > 0:
                    alive = np.flatnonzero(self.lifetimes[i, :, 1])
                    if self.lifetimes[i, alive, 1] == 1: #Cluster dissociates
                        ax.plot(times[alive[0]], volumes[-1], color=color, linewidth=2, marker="*", markersize=15)
                    elif self.lifetimes[i, alive, 1] == 2: #Cluster joins another small cluster
                        ax.plot(times[alive[0]], volumes[-1], color=color, linewidth=2, marker="o", markersize=15)
                    else:  #Cluster joins largest cluster
                        ax.plot(times[alive[0]], volumes[-1], color=color, linewidth=2, marker="P", markersize=15)

            legend_elements = [Line2D([0], [0], marker="*", color='w', markerfacecolor='b', markersize=15, label='Cluster Fully Unpaired'),
                               Line2D([0], [0], marker='P', color='w', markerfacecolor='g', markersize=15, label='Cluster Joins Largest Cluster'),
                               Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=15, label='Cluster Joins Small Cluster')]
            ax.legend(handles=legend_elements, prop={'size': 18})


    def lifetimePDF(self):
        """This method creates a 3D PDF of cluster lifetimes and volumes. Based on ArtifexR's response at
        https://stackoverflow.com/questions/44895117/colormap-for-3d-bar-plot-in-matplotlib-applied-to-every-bar"""
        take = self.age[:, 1] == 1
        x = self.age[take, 0] #Isolate clusters which do not merge
        y = self.aveV[take] / self.eta ** 3

        fig = plt.figure(figsize = (20, 20))
        ax = fig.add_subplot(111, projection='3d')

        hist, xedges, yedges = np.histogram2d(x, y, bins=(20, 20), density=True)
        xpos, ypos = np.meshgrid(xedges[:-1] + xedges[1:], yedges[:-1] + yedges[1:])

        xpos = xpos.flatten() / 2.
        ypos = ypos.flatten() / 2.
        zpos = np.zeros_like(xpos)

        dx = xedges[1] - xedges[0]
        dy = yedges[1] - yedges[0]
        dz = hist.flatten()

        cmap = cm.get_cmap('autumn')
        max_height = np.max(dz)
        min_height = np.min(dz)
        rgba = [cmap((k - min_height) / max_height) for k in dz]

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
        ax.tick_params(axis='both', labelsize=15)
        plt.title("PDF of Cluster Volume and Lifetime", fontsize=27)
        plt.xlabel("\nCluster Lifetime " + r'$\frac{t}{t_\eta}$ [-]', fontsize=24, linespacing=3.2)
        plt.ylabel("\nCluster Volume " + r'$\frac{V}{\eta^3}$ [-]', fontsize=24, linespacing=3.2)

    def causeofdeathPDFs(self):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(16, 11)
        ax.set_xlabel(r'$\frac{V}{\eta^3}$ [-]', fontsize=24)
        ax.grid(True)
        ax.set_ylabel('PDF [-]', fontsize=24)
        ax.tick_params(axis='both', labelsize=15)
        ax.set_title('PDF of Cluster Ends with Volume', fontsize=20)
        fig.tight_layout()
        cause = ['Cluster Survives', 'Cluster Unpaired', 'Cluster Merges with Small Cluster', 'Cluster Merges with Largest Cluster']

        for i in range(4):
            take = self.age[:, 1] == i
            hist, bins = np.histogram(self.aveV[take] / self.eta ** 3, bins=20, density=True)
            bins = (bins[1:] + bins[:-1]) / 2
            ax.plot(bins, hist, label=cause[i], linewidth=3)

        plt.legend(fontsize=18)
        plt.grid()




    def trackCluster(self, c=1, plot=True):
        """This method tracks a certain cluster c, by observing the cluster to which its majority of skeleton points
        transform into, and plotting the number of skeleton points and volume of such cluster at each time instant"""
        volumes = np.zeros(len(self.pairs) + 1)
        skel = np.zeros(len(self.pairs) + 1)
        cluster = c
        labelSequence = [c]

        for i, p in enumerate(self.pairs):
            volumes[i] = self.volumes[i][cluster - 1]
            skel[i] = np.sum(self.transitions[i][cluster - 1, :])
            cluster = np.argmax(self.transitions[i][cluster - 1, 1:]) + 1
            labelSequence.append(cluster)
            if i == len(self.pairs) - 1:
                volumes[i + 1] = self.volumes[i + 1][cluster - 1]
                skel[i + 1] = sum(self.trackers[i].labels[1] == cluster)

        if plot:
            fig, ax1 = plt.subplots()
            fig.set_size_inches(16, 8)
            color = 'tab:red'
            ax1.set_xlabel('Time ' + r'$\frac{t}{t_\eta}$ [-]', fontsize=24)
            plt.xticks(fontsize=15)
            plt.grid()
            ax1.set_ylabel("Cluster Volume " + r'$\frac{V}{\eta^3}$ [-]', color=color, fontsize=24)
            ax1.plot(np.array(self.times) / self.tK, volumes / self.eta ** 3, color=color)
            ax1.tick_params(axis='y', labelcolor=color, labelsize=15)
            ax1.set_title('Evolution of Volume and Number of Skeleton Points of Cluster ' + str(c), fontsize=20)

            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Member Skeleton Points [-]', color=color, fontsize=18)
            ax2.plot(np.array(self.times) / self.tK, skel, color=color)
            ax2.tick_params(axis='y', labelcolor=color, labelsize=15)
        return self.times, volumes, skel, labelSequence

# #pairs = [[0, 100], [100, 200]] #, [200, 300], [300, 400], [400, 500]]
# # pairs = [[0, 500], [500, 1000], [1000, 1500], [1500, 2000]]
# pairs = np.transpose(np.vstack((np.arange(0, 4100, 100), np.arange(100, 4100 + 100, 100))))
# tt = temporalTrackerGlobal(pairs, root='./Data/v78/t_', plot=False)
# tt.trackVolumes()
# tt.lifetimePDF()
# tt.causeofdeathPDFs()
# tt.trackCluster()
# plt.show()