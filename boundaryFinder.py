import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import time
from skeletonPlot import skeletonPlot

# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is in charge of determining, for a set of isolated particle clusters within different
# snapshots in Z, the cluster particles which form the boundary of the cluster. In order to do so, the main class
# "boundaryFinder" invokes, for every snapshot in Z, the class "boundaryTraveller", the latter obtaining a closed particle
# boundary for each of the clusters in the dataset.
# Álvaro Tomás Gil - UIUC 2020

class boundaryFinder:
    """This is the main class of the script, in charge of calling the secondary class for every Z value in the dataset,
    performing an additional polishing of the obtained cluster boundaries, and plotting the resulting boundaries. For the
    input:
    data: 2D Array containing the 3D positions of all particles, arranged into a discrete set of Z levels, to which the
    particle positions have been normally projected
    labels: List which labels each of the elements in data based on their DBSCAN cluster label. In case this label is -1,
    the referenced element corresponds to a void particle
    eps: Float defining the eps value based on which clustering analysis with DBSCAN has been performed. This is used as a
    reference length in this method.
    folder: folder in which to save resulting plots"""
    def __init__(self, data, labels, eps, folder=''):
        self.data = data
        self.labels = labels
        self.eps = eps
        self.folder = folder
        self.zs = np.unique(self.data[:, 2])
        self.fakeSkel = np.array([0, 0, 0])


    def bound2dUnsupervised(self, z, k=0.5, dirs=6):
        """This method is in charge of invoking the secondary class in order to perform the boundary analysis of the clusters
        for which Z is the same as the specified value."""
        self.bT = boundaryTraveller(z, self.data, self.labels, self.eps, k=k, dirs=dirs)
        boundary, plane, cluster = self.bT.run()
        return boundary, plane, cluster

    def sweep(self, runs=1, plot3d=True, plot=[], k=0.5, dirs=6):
        """This method is central to the current class, as it sweeps for the different Z values existing in the dataset,
        and calls the cluster boundary determination. Moreover, this method calls the method in charge of polishing
        the resulting cluster boundaries and plots the final results."""
        start = time.time()
        print('Sweep along Z of Boundary Finder')
        self.boundaries = []
        self.bound3d = np.array([0, 0, 0])
        self.clusters = np.array([0, 0, 0])

        for i, z in enumerate(self.zs):
            print('Z = ' + str(round(z, 3)))
            complete = []
            for j in range(runs):
                boundary, plane, cluster = self.bound2dUnsupervised(z, k, dirs)
                # Fake skel is an array whose exclusive function is for plotting the resulting cluster boundaries
                self.fakeSkel = np.vstack((self.fakeSkel, cluster[0, :]))
                complete = complete + boundary
            boundary = list(set(complete))
            self.boundaries.append(boundary)

            #bound3d is the 2d array of all cluster boundary particle positions
            self.bound3d = np.vstack((self.bound3d, plane[boundary, :]))
            self.clusters = np.vstack((self.clusters, cluster))


        print('Execution Time: ' + str(round(time.time() - start, 3)))

        self.bound3d = self.bound3d[1:, :]
        self.clusters = self.clusters[1:, :]

        self.polishBoundaries()

        if plot3d:
            self.plotAll()

        for i in plot:
            z = self.zs[i]
            self.boundaryPlot(z)

    def polishBoundaries(self, p=0.2, kb=0.3):
        """This method polishes the obtained cluster boundaries by deleting boundary particles with less than a fraction
        p of the average number of neighboring boundary particles within a neighborhood of kb*epsilon"""
        self.nbrs = NearestNeighbors(radius=kb * self.eps, algorithm='auto')
        self.nbrs.fit(self.bound3d)
        dis, ind = self.nbrs.radius_neighbors()
        neighs = []
        for i, v in enumerate(ind):
            neighs.append(len(v))

        ave = np.mean(neighs)
        keep = [i for i in range(len(neighs)) if neighs[i] > p * ave]
        print('Ave Neighs: ' + str(ave) + '. Min Neighs: ' + str(np.min(neighs)) + '. Killed: ' + str(
            len(self.bound3d) - len(keep)))
        self.bound3d = self.bound3d[keep, :]

    def boundaryPlot(self, z):
        """Given an array of boundary particle positions 'boundary', this method plots these results. """
        take = self.bound3d[:, 2] == z
        boundary = self.bound3d[take, :]
        take = self.clusters[:, 2] == z
        cluster = self.clusters[take, :]
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 9.5)
        ax.set_title('Boundary Particles of Cluster for Z = ' + '{0:03f}'.format(boundary[0, 2]), fontsize=23)
        ax.set_xlabel('x [m]', fontsize=18)
        ax.set_ylabel('y [m]', fontsize=18)
        plt.axis('equal')
        ax.scatter(boundary[:, 0], boundary[:, 1], s=0.8, c='red')
        ax.scatter(cluster[:, 0], cluster[:, 1], s=0.8, c='blue')

    def plotAll(self):
        """This method invokes the plotting of cluster boundaries for all of the considered Z levels"""
        self.fakeSkel = self.fakeSkel[1:]
        sP = skeletonPlot(self.bound3d, self.fakeSkel, folder=self.folder)
        sP.snapPlot(title='Lagrangian Boundary')


class boundaryTraveller:
    """For a particular Z level, this class is in charge of determining the cluster particles which represent cluster
    boundaries. For the cluster particles contained in less dense regions, the algorithm proceeds by measuring the angles
    between successive particle-neighboring particle vectors. If any of these angles for a particular cluster particle
    is larger than a specified threshold value, the cluster particle is considered to be a boundary, and the neighboring
    particles adjacent to such gap are stored for later analysis. For the inputs:
    z: Float describing the z value of data to analyze
    data: 2D Array containing the 3D positions of all particles, arranged into a discrete set of Z levels, to which the
    particle positions have been normally projected
    labels: List which labels each of the elements in data based on their DBSCAN cluster label. In case this label is -1,
    the referenced element corresponds to a void particle
    radius: Float defining the reference length to employ for the neighborhood radius of each particle under analysis
    dirs: Dividing 2*Pi by this float, one obtains the minimum separation angle between adjacent neighboring particles
    for the particle under analysis to be considered as a boundary
    k: Multiplied by radius, this float defines the neighborhood radius based on which the neighbors of each particle are
    analyzed
    dense: Proportion of the maximum number density for a particle to be considered by this algorithm
    """
    def __init__(self, z, data, labels, radius=0.001, dirs=6, k=0.5, dense=0.75):
        self.z = z
        self.dir = dirs
        self.plane = data[data[:, 2] == z, :2]
        self.labels2d = labels[data[:, 2] == z]
        self.cluster = self.plane[self.labels2d != -1, :] #only cluster particles
        self.clustInd = np.ndarray.flatten(np.argwhere(self.labels2d != -1))  # references plane

        self.nbrs = NearestNeighbors(radius=radius * k, algorithm='auto').fit(self.cluster)
        self.distances, self.indices = self.nbrs.radius_neighbors(self.cluster)

        self.N = []
        for i in self.indices:
            self.N.append(len(i))

        self.notdense = [i for i in range(len(self.N)) if self.N[i] < dense*np.max(self.N)] #extracts less dense particles

        self.taken = np.array(
            [])  # Taken and Queue both index Cluster and ClustInd, whileas ClustInd references self.plane
        self.queue = np.array([])
        self.boundary = []  # References self.plane

        self.theta = [(i - 1) * 2 * np.pi / self.dir for i in range(1, self.dir + 1)]
        self.ri = np.array([[np.cos(th), np.sin(th)] for th in self.theta])

        self.test = np.random.randint(0, len(self.cluster))

        print('	boundaryTraveller: dirs = ' + str(dirs) + '; k = ' + str(k) + ';')

    def nextInLine(self):
        """This method is in charge of determine, at each iteration of the algorithm, which particle to analyze next. If
        the queue of particles is empty, a particle from the less denser group of particles is randomly sampled."""
        if len(self.queue) > 0:
            take = self.queue[0]
            self.queue = np.delete(self.queue, 0)
        else:
            poss = np.setdiff1d(self.notdense, self.taken)
            take = poss[np.random.randint(0, len(poss))]

        self.taken = np.append(self.taken, take)
        return int(take)

    def substractAngles(self, this, dumping=False):
        """Given a particle indexed by 'this', this method examines all of its neighboring particles and measures the
        angles between adjacent particle-neighboring particle vectors. All neighboring particles for which the angle
        of separation is greater than the specified threshold are stored in 'adj' to be added to the queue. If dumping
        is allowed, all the other neighboring particles are excluded from further processing by the algorithm via 'dump'."""
        pos = self.cluster[this, :]
        neighs = self.indices[this]
        neighs = neighs[neighs != this]
        vecs = np.array([0, 0])
        angs = np.array([])

        for i in neighs:
            that = self.cluster[i, :]
            mag = np.linalg.norm(that - pos)
            if mag > 0:
                vecs = np.vstack((vecs, (that - pos) / mag))
                theta = np.arctan2(vecs[-1, 1], vecs[-1, 0])
                if theta < 0:
                    theta += 2 * np.pi
                angs = np.append(angs, theta)

        if len(angs) > 0:
            vecs = vecs[1:, :]
            sortI = np.argsort(angs)
            angs = np.sort(angs)

            sortI = np.append(sortI, sortI[0])
            angs = np.append(angs, angs[0] + 2 * np.pi)

            delta = np.diff(angs)
            gaps = np.flatnonzero(delta >= 2 * np.pi / self.dir)
            adjacent = np.concatenate((gaps, gaps + 1))
            adj = np.unique(neighs[sortI[adjacent]])

            if dumping:
                dump = [i for i in neighs if i not in adj]
            else:
                dump = []

        # print('Newline:			', angs*180/np.pi, delta*180/np.pi, adjacent, adj)

        else:
            adj = []
            dump = []

        return adj, dump

    def processPoint(self, this):
        """For a cluster particle indexed via 'this', this method extracts whether this particle can be considered as a
        cluster boundary particle. It also stores relevant neighboring particles of this particle in the particle
        queue for further analysis, and all dumped neighboring particles in the list of taken particles to exclude them
        from further analysis."""
        take, discard = self.substractAngles(this)

        if len(take) > 0:
            self.boundary.append(int(self.clustInd[this]))
            add2queue = [v for v in take if v not in self.taken]
            self.queue = np.append(self.queue, add2queue)

        if len(discard) > 0:
            add2taken = [v for v in discard if v not in self.taken]
            self.taken = np.append(self.taken, add2taken)

    def run(self):
        """This method executes the boundary determining algorithm, by sampling a cluster particle as per
        nextInLine and processing such particle with processPoint. What results is boundary, a list of indexes of
        boundary particles referencing plane, plane, an array of particle positions for the current Z level, and cluster,
        an array of cluster particle positions for the current Z level."""
        while len(self.taken) < len(self.notdense):
            this = self.nextInLine()
            # print('Next Point in Line: ', this)
            self.processPoint(this)
        # print(self.boundary.shape, self.plane.shape, self.cluster.shape)
        self.plane = np.hstack((self.plane, self.z * np.ones((len(self.plane), 1))))
        self.cluster = self.plane[np.setdiff1d(self.clustInd, self.boundary), :]
        return self.boundary, self.plane, self.cluster

