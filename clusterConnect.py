import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
from skeletonPlot import skeletonPlot

# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: Based on the resulting Brute Skeleton representation of several DBSCAN clusters within different levels of Z,
# this script is in charge of examining the connectivity of every brute skeleton point with all others, by determining whether
# a straight line between both skeleton points collides with a cluster boundary or not. In order to examine connectivities along
# different levels of Z, the projection of the skeleton point to the other level of Z is used instead of the original skeleton point.
# Álvaro Tomás Gil - UIUC 2020

class skeletonConnect:
    """This is the main class of the script. Based on the resulting Brute Skeleton representation of several DBSCAN clusters within different levels of Z,
    this script is in charge of examining the connectivity of every brute skeleton point with all others, by determining whether
    a straight line between both skeleton points collides with a cluster boundary or not. In order to examine connectivities along
    different levels of Z, the projection of the skeleton point to the other level of Z is used instead of the original skeleton point.
    For the inputs:
    skel: 2D array of 3D skeleton point positions
    bound3d: 2D array of 3D boundary particle positions
    eta: float, Kolmogorov length scale, used as a reference length
    kb: float, Determines the neighborhood distance with which to extract relevant skeleton positions, by multiplying it with eta
    kc: float, Determines the neighborhood distance under which to assume that a trajectory between two skeleton points has collided
    with a cluster boundary, by multiplying it with eta
    ks: float, Determines the separation distance between adjacent points defining the trajectory connecting two skeleton particles,
    by multiplying it by eta
    folder: folder in which to save resulting plots"""
    def __init__(self, skel, bound3d, eta=70.59*10**-6, kb=106.25, kc=2.12, ks=0.42, folder=''):
        self.skel = skel
        self.bound3d = bound3d
        self.eta = eta
        self.folder = folder

        self.ds = ks * eta
        self.zs = np.unique(bound3d[:, 2])
        self.dz = np.abs(self.zs[1] - self.zs[0]) / 4

        #Neighborhood of skeleton particles
        self.nbrs = NearestNeighbors(radius=kb * eta, algorithm='auto').fit(skel)

        #Neighborhood of boundary particles, for collisions
        self.collision = NearestNeighbors(radius=kc * eta, algorithm='auto').fit(bound3d)

        #For every skeleton particle, this list contains a list of connected skeleton particles
        self.connexions = [[] for i in range(len(self.skel))]
        self.trajs = np.array([0, 0, 0])

    def run(self):
        """Main method of the class, in charge of its execution."""
        print('Determination of Skeleton Connectivity')
        start = time.time()
        for i in range(len(self.skel)):
            connexions, trajs = self.forSkelP(i)
            if trajs is not None:
                self.connexions[i] = connexions
                self.trajs = np.vstack((self.trajs, trajs))

        self.trajs = self.trajs[1:, :]
        self.mutualizer()
        self.labeller()
        self.plotTrajs()
        print('Execution Time: ' + str(round(time.time() - start, 3)))

    def forSkelP(self, i):
        """For a single skeleton point, defined by index i, this method examines whether it can be connected to nearby skeleton
        particles with straight trajectories, taking into account the point's projections on the two adjacent levels of Z"""
        r = self.skel[i, :2]
        thisZ = self.skel[i, 2]
        thisZi = np.argmin(np.abs(self.zs - thisZ))

        #allZ samples adjacent Z levels as well as the Z level of skeleton point i
        allZ = self.zs[max(0, thisZi - 1):min(len(self.zs), thisZi + 1) + 1]
        connexions = []
        trajs = np.array([0, 0, 0])

        for z in allZ:
            r1 = np.append(r, z)
            conn, traj = self.findNeigh(r1, thisZ)
            if traj is not None:
                connexions = connexions + conn
                trajs = np.vstack((trajs, traj))

        if len(trajs.shape) == 2:
            trajs = trajs[1:, :]
        else:
            trajs = None
        return connexions, trajs

    def findNeigh(self, r, originalZ):
        """Given a skeleton point position r, this method finds the nearby skeleton positions which can be connected
        by means of a straight trajectory, and outputs the indices of the connected neighbors as well as the connecting
        trajectory points."""
        thisZ = r[2]
        distance, indexes = self.nbrs.radius_neighbors(np.reshape(r, (1, -1)))
        indexes = indexes[0]
        distance = distance[0]
        indexes = indexes[distance > 1e-10]

        take = ((self.skel[indexes, 2] >= thisZ - self.dz) & (self.skel[indexes, 2] <= thisZ + self.dz))
        indexes = indexes[take]
        connexions = []
        trajs = np.array([0, 0, 0])

        for i in indexes:
            r2 = self.skel[i, :]
            conn, traj = self.areConnected(r, r2, originalZ)

            if conn:
                connexions.append(i)
                trajs = np.vstack((trajs, traj[::5, :]))

        if len(trajs.shape) == 2:
            trajs = trajs[1:, :]
        else:
            trajs = None

        return connexions, trajs

    def areConnected(self, r1, r2, originalZ):
        """Given two positions r1 and r2, this method is in charge of determining whether a straight line between both
        points is at any point closer than kc * eta to a cluster boundary. If this is not so, both points are said to be connected,
        (connected = True), and the trajectory between both is returned. In the case in which r1 is not an actual skeleton
        point but rather a projection into an adjacent level in Z, this method returns a different trajectory than the one
        employed to study connectivities. This trajectory is the straight line between the original skeleton point at its original
        Z and the second skeleton point."""
        thisZ = r1[2]
        connected = True
        traj = self.defTraj(r1, r2)
        indexes = self.collision.radius_neighbors(traj, return_distance=False)
        for i, v in enumerate(traj):
            ind = indexes[i]
            take = ((self.bound3d[ind, 2] >= thisZ - self.dz) & (self.bound3d[ind, 2] <= thisZ + self.dz))
            ind = ind[take]

            if len(ind) > 0:
                connected = False
                break

        if connected and originalZ != thisZ:
            r1[2] = originalZ
            traj = self.defTraj(r1, r2)

        return connected, traj

    def defTraj(self, r1, r2):
        """This simple method outputs the points belonging to the straight trajectory between points r1 and r2"""
        numPts = np.ceil(np.linalg.norm(r1 - r2) / self.ds)
        traj = np.linspace(r1, r2, numPts)
        return traj

    def plotTrajs(self):
        """This method invokes class skeletonPlot to define a descriptive plot of the determined connections"""
        self.plot = skeletonPlot(self.bound3d, self.trajs, skel2=self.skel, labels=self.labels, folder=self.folder)
        # self.plot.messyPlot(title = 'Connectivity of Skeleton Points', labelS = 'Connexions')
        self.plot.snapPlot(title='Connectivity of Skeleton Points', labelS='Connexions')

    def mutualizer(self):
        """Based on a list of connexions, this method makes sure that if i is included in the connexions of j, so is j included
        in the connexions of i."""
        for i, c in enumerate(self.connexions):
            for j in c:
                if i not in self.connexions[j]:
                    self.connexions[j].append(i)

    def labeller(self):
        """Based on the defined connexions, this method is in charge of assigning a label to each skeleton particle, such that
        connected skeleton points have the same label, but only the lowest possible labels are assigned."""
        self.labels = [0 for i in range(len(self.skel))]
        print('	Assignment of Cluster Labels')

        for _ in range(1):
            for i, c in enumerate(self.connexions):
                if self.labels[i] == 0:
                    self.labels[i] = max(self.labels) + 1
                # print('SP: ' + str(i) + '. Connexions: ' + str(c) + '. Li: ' + str(self.labels[i]))
                for j in c:
                    # print('SP: ' + str(j) + '. Lj: ' + str(self.labels[j]))
                    l = self.labels[j]
                    current = self.labels[i]
                    if l == 0:
                        self.labels[j] = current
                    elif l < current and l > 0:
                        self.mergeLabels(l, current)
                    elif l > current:
                        self.mergeLabels(current, l)
            un, counts = np.unique(self.labels, return_counts=True)
            res = [(un[i], counts[i]) for i in range(len(un))]
            print('	Cluster Labels and Frequencies: ', res)

    def mergeLabels(self, winner, loser):
        """In the case in which two already labelled skeleton points are found to be connected, this method determines
        which label should be propagated (winner), and which label (loser) should be substituted."""
        # print('Winner: ' + str(winner) + '. Loser: ' + str(loser))

        # for i, l in enumerate(self.labels):
        #     if l == loser:
        #         self.labels[i] = winner
        #
        # if len(np.unique(self.labels)) > 1:
        #     self.new = self.labels.copy()
        #     for i, l in enumerate(np.unique(self.labels)):
        #         for j, k in enumerate(self.labels):
        #             if k == l:
        #                 self.new[j] = i
        #     self.labels = self.new.copy()
        l1 = winner
        l2 = loser

        if l1 != l2:
            winner = min(l1, l2)
            loser = max(l1, l2)
            loserN = 0
            superiorN = 0
            for i,l in enumerate(self.labels):
                if l == loser:
                    loserN += 1
                    self.labels[i] = winner
                if l > loser:
                    superiorN += 1
                    self.labels[i] = l - 1