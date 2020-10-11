import numpy as np
import matplotlib.pyplot as plt
from clusterPlot import clusterPlot
from skeletonPlot import skeletonPlot
from sklearn.neighbors import NearestNeighbors
import time

# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is designed to analyze a DBSCAN-labelled array of particle positions in order to discretize its 2D
# domain in an Eulerian way, determine cluster boundaries based on which of the Eulerian cells are populated by particles labelled
# by DBSCAN as cluster particles, compute the area associated to each DBSCAN label, and determine an Eulerian skeleton with Eulerian
# cell centers in areas labelled as cluster areas by DBSCAN.
# Álvaro Tomás Gil - UIUC 2020


class eulerianAnalysis:
    """This class discretizes the 2D domain and determines which Eulerian cells are populated by cluster particles. It also determines
    cluster boundaries based on populated cells which are neighbors to non-populated cells.
    cluster: 3-column array of cluster particles, sorted by 2D snapshots in Z
    labels: Array of labels, labeling each element in cluster according to a previous DBSCAN analysis.
    eps: epsilon for the previous DBSCAN analysis, used as a reference length
    kx, ky: Constants determining Eulerian cell sizes in X and in Y when multiplied by eps
    boundMethod: Specifies the method to be used to determine cluster boundaries
    lagBounds: Array of cluster boundaries obtained by "boundaryTraveller", to use as a reference for Eulerian cell sizes
    if kx and ky are None
    folder: folder in which to save resulting plots"""
    def __init__(self, cluster, labels, eps, kx=0.2, ky=0.2, boundMethod='basic', lagBounds=None, folder=''):
        print('Eulerian Analysis of Boundaries and Areas')
        cluster = cluster[labels != -1, :]
        self.clusters = cluster
        labels = labels[labels != -1]
        self.folder = folder
        self.cluster = cluster
        self.labels = labels
        self.eps = eps
        self.method = boundMethod

        self.reduce = cluster
        self.redL = labels

        if kx is None or ky is None:
            kx = self.determineDims(lagBounds=lagBounds)
            ky = kx

        self.dx = kx * eps
        self.dy = ky * eps
        self.zs = np.unique(cluster[:, 2])

        self.xrange = [np.min(cluster[:, 0]), np.max(cluster[:, 0])]
        self.yrange = [np.min(cluster[:, 1]), np.max(cluster[:, 1])]
        self.xn = int(np.floor((self.xrange[1] - self.xrange[0]) / self.dx))
        self.yn = int(np.floor((self.yrange[1] - self.yrange[0]) / self.dy))

        self.x = np.linspace(self.xrange[0], self.xrange[1], self.xn)
        self.y = np.linspace(self.yrange[0], self.yrange[1], self.yn)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.a = self.dx*self.dy

        self.xc = np.linspace(self.dx/2 + self.x[0], self.x[-1] - self.dx/2, self.xn - 1)
        self.yc = np.linspace(self.dy / 2 + self.y[0], self.y[-1] - self.dy / 2, self.yn - 1)
        self.isfull = -1*np.ones((self.xn - 1, self.yn - 1, len(self.zs)))
        self.Np = np.zeros((self.xn - 1, self.yn - 1, len(self.zs)))

        #rc: 2D array containing the Eulerian cell centers for all snapshots in Z
        #lc: Label array defining whether a cluster populates an Eulerian cell, and in such case, of which DBSCAN label
        self.rc = np.zeros(((self.xn - 1) * (self.yn - 1) * len(self.zs), 3))
        self.lc = -1*np.ones(((self.xn - 1) * (self.yn - 1) * len(self.zs)))
        self.bound3d = np.array([0, 0, 0])
        self.fakeSkel = np.array([0, 0, 0])
        self.areas = [0 for _ in self.zs]
        self.labelcount = [[] for _ in self.zs]

    def determineDims(self, lagBounds):
        """This method is in charge of assigning Eulerian cell dimensions based on the average separation between an already existing
        cluster boundary"""
        if lagBounds is None:
            lagrangian = np.load('lagrangianBound3d.npy')
        else:
            lagrangian = lagBounds
        nbrs = NearestNeighbors(n_neighbors=1).fit(lagrangian)
        dist, _ = nbrs.kneighbors()
        dist = np.ndarray.flatten(dist)
        k = np.mean(dist) / self.eps
        print(' Based on the mean distance between boundary particles in the Lagrangian boundary, k = ' + str(round(k, 3)))
        return k

    def testSnapshot(self):
        """This method is in charge of testing the whole routine for a single Z snapshot"""
        i = 0
        take = np.arange(len(self.cluster[:, 2]))[self.cluster[:, 2] == self.zs[i]]
        self.fakeSkel = np.vstack((self.fakeSkel, self.cluster[take[0]]))
        self.fullI = np.array([0, 0])
        self.filloccupancies(i)
        return (len(self.fullI) - 1)/((self.xn - 1)*(self.yn - 1))

    def run(self):
        """Generic run command. The output of this command is a 2D array containing the coordinates of all cluster boundary points. (bound3d)"""
        start = time.time()
        for k,z in enumerate(self.zs):
            self.forZ(k)
            print(' Cells per DBSCAN label for Z = ' + str(round(z, 3)) + ':')
            for key in sorted(self.labelcount[k]):
                print("     %s: %s" % (key, self.labelcount[k][key]))
        self.fakeSkel = self.fakeSkel[1:, :]

        self.plotisfull()
        if self.method is not None:
            self.bound3d = self.bound3d[1:, :]
            self.plotBoundary()
        print(' Average Number of Particles per Cell = ' + str(round(np.mean(self.Np), 3)))
        print(' Average Number of Particles per Cell in Populated Cells = ' + str(round(np.mean(self.Np[self.Np > 0]), 3)))
        print('Execution Time: ' + str(round(time.time() - start, 3)))


    def forZ(self, i):
        """Execution for every snapshot in Z"""
        take = np.arange(len(self.cluster[:, 2]))[self.cluster[:, 2] == self.zs[i]]
        self.fakeSkel = np.vstack((self.fakeSkel, self.cluster[take[0]]))
        self.fullI = np.array([0, 0])
        self.filloccupancies(i)

        if self.method == 'skimage':
            self.skimageBound(i)
        elif self.method is not None:
            self.seeneighbors(i)

    def filloccupancies(self, k):
        """This method is in charge of sweeping along the 2D domain, determining whether each Eulerian cell is populated by
        cluster particles or not. This result is stored in:
        isfull: This is a 3D array of indices corresponding to the Eulerian grid (column, row, Zsnapshot) contains the label
        assigned to every Eulerian cell, which is either the DBSCAN label of the cluster particles within it, or -1
        rc and lc: These two 2D arrays both display what isfull does without employing 3D array schematics"""
        z = self.zs[k]
        areas = {}
        labelcount = {}
        tz = np.arange(len(self.reduce[:, 2]))[self.reduce[:, 2] == z]
        rz, self.reduce = self.newanddelete(tz, self.reduce)
        lz, self.redL = self.newanddelete(tz, self.redL)

        for i,x in enumerate(self.xc):
            x1 = self.x[i]
            x2 = self.x[i + 1]
            tx = np.arange(len(rz))[((rz[:, 0] > x1) & (rz[:, 0] < x2))]
            rx, rz = self.newanddelete(tx, rz)
            lx, lz = self.newanddelete(tx, lz)
            for j,y in enumerate(self.yc):
                y1 = self.y[j]
                y2 = self.y[j + 1]
                ty = np.arange(len(rx))[((rx[:, 1] > y1) & (rx[:, 1] < y2))]
                ry, rx = self.newanddelete(ty, rx)
                ly, lx = self.newanddelete(ty, lx)
                self.rc[self.longindex(i, j, k), :] = np.array([x, y, z])
                if len(ty) > 0:
                    winner = np.random.choice(ly, 1)[0]
                    self.isfull[i, j, k] = winner
                    self.Np[i, j, k] = len(ty)
                    self.lc[self.longindex(i, j, k)] = winner
                    self.fullI = np.vstack((self.fullI, np.array([i, j])))

                    if winner not in areas.keys():
                        areas[winner] = self.a
                        labelcount[winner] = 1
                    else:
                        areas[winner] += self.a
                        labelcount[winner] += 1
                else:
                    self.lc[self.longindex(i, j, k)] = -1
        self.areas[k] = areas
        self.labelcount[k] = labelcount

    def seeneighbors(self, k):
        """This method is in charge of, for every populated Eulerian cell, determining if one of the directly neighboring cells
        are populated by cluster particles."""
        self.fullI = self.fullI[1:, :]
        for indices in self.fullI:
            i = indices[0]
            j = indices[1]
            if j + 1 > self.yn - 2:
                right = True
            else:
                right = self.isfull[i, j + 1, k] == -1

            if j - 1 < 0:
                left = True
            else:
                left = self.isfull[i, j - 1, k] == -1

            if i - 1 < 0:
                up = True
            else:
                up = self.isfull[i - 1, j, k] == -1

            if i + 1 > self.xn - 2:
                down = True
            else:
                down = self.isfull[i + 1, j, k] == -1

            if any([up, down, left, right]):
                self.bound3d = np.vstack((self.bound3d, self.rc[self.longindex(i, j, k), :]))

    def skimageBound(self, k):
        """Boundary determination employing skimage, where the input is a black-and-white image corresponding to the Eulerian grid"""
        from skimage import measure
        self.fullI = self.fullI[1:, :]
        canvas = np.zeros(self.isfull.shape[:2])

        for indices in self.fullI:
            i = indices[0]
            j = indices[1]
            canvas[i, j] = 1

        contours = measure.find_contours(canvas, 0)
        if type(contours) is not list:
            contours = [contours]

        for c in contours:
            for r in c:
                i = int(r[0])
                j = int(r[1])
                self.bound3d = np.vstack((self.bound3d, self.rc[self.longindex(i, j, k), :]))


    def longindex(self, i, j, k):
        """Converts three indices used in the 3D array to two indices used in 2D array equivalent"""
        r = j + (self.yn - 1)*i + (self.yn - 1)*(self.xn - 1)*k
        return r

    @staticmethod
    def newanddelete(index, array):
        """Method in charge of extracting and deleting an element from an array"""
        new = array[index]
        array = np.delete(array, index, axis=0)
        return new, array

    def plotisfull(self):
        """Plotting of the Eulerian grid represented in isfull"""
        cP = clusterPlot(self.rc, self.lc, self.folder)
        cP.plotAll('IsFull')

    def plotBoundary(self):
        """Plotting of the resulting cluster boundaries"""
        sP = skeletonPlot(self.bound3d, self.fakeSkel, folder=self.folder)
        sP.snapPlot(title='Eulerian Boundary')

class eulerianSkeleton:
    """This class is in charge of creating a brute skeleton defining the cluster's shape based on Eulerian cell centers which have,
    not only a sufficient separation from any predefined cluster boundary, but also a separation from all other skeleton particles
    within the cluster. The input to this class is:
    rc and lc: Eulerian cell centers and labels, as per generated in the previous class
    bound3d: 2D array of coordinates of cluster boundary points
    eta: Kolmogorov length scale, used as a reference length
    ks: Defines, when multiplied by eta, the minimum separation between skeleton particles
    kc: Defines, when multiplied by eta, the minimum separation to hold between skeleton particles and boundary particles
    folder: folder in which to save resulting plots
    """
    def __init__(self, rc, lc, bound3d, eta=70.59*10**-6, ks=17, kc=3.2, folder=''):
        self.rc = rc
        self.lc = lc
        self.bound3d = bound3d
        self.eta = eta
        self.dx = abs(rc[0, 1] - rc[1, 1])
        self.ds = eta*ks
        self.crash = kc*eta
        self.folder = folder

        self.zs = np.unique(self.rc[:, 2])
        self.skel = np.array([0, 0, 0])

    def run(self, title='EulerianSkeleton'):
        print('Topological Skeletonization of Clusters')
        start = time.time()
        for z in self.zs:
            print(' Z = ' + str(round(z, 3)))
            self.forZ(z)

        self.skel = self.skel[1:, :]
        print('Execution Time: ' + str(round(time.time() - start, 3)))
        self.plotSkel(title)

    def forZ(self, z):
        releC = ((self.rc[:, 2] == z) & (self.lc != -1))
        self.clusters = self.rc[releC, :]
        self.labels = self.lc[releC]

        releB = self.bound3d[:, 2] == z
        self.boundary = self.bound3d[releB, :]
        self.nbrs = NearestNeighbors(n_neighbors=1).fit(self.boundary)

        for l in np.unique(self.labels):
            self.forCluster(l)


    def forCluster(self, l, ksample=0.3):
        """This method analyzes a single DBSCAN-labelled cluster in a single 2D snapshot in Z, and determines a set of Eulerian cell
        centers which are the farthest away possible from all boundary particles and are separated enough from each other."""
        cluster = self.clusters[self.labels == l]
        cluster = cluster[np.random.choice(np.arange(len(cluster)), max(int(ksample*len(cluster)), 1))] #Sample only a fraction ksample of cluster cell centers
        cluNbrs = NearestNeighbors(radius=50*self.ds).fit(cluster)
        score = np.array([])
        for i,r in enumerate(cluster):
            minD, _ = self.nbrs.kneighbors(np.reshape(r, (1, -1)))
            score = np.append(score, minD)

        taken = [np.argmax(score)] #Array defining which Eulerian cell centers already define the skelleton
        done = False
        it = 0
        while not done:
            it += 1
            #Every taken cell has a best ranked, far enough, not taken cell
            distances, indices = cluNbrs.radius_neighbors(cluster[taken, :])
            far = []
            for i, row in enumerate(distances):
                farenough = indices[i][row > self.ds] #Which Eulerian cell centers are far enough from the current skeleton particle?
                farvirgin = [x for x in farenough if x not in taken] #Which of these are not already taken?
                farvirgin = [f for f in farvirgin if score[f] > self.crash] #Which of these are not too close to cluster boundaries?
                if len(far) == 0:
                    far = farvirgin
                else:
                    far = [f for f in far if f in farvirgin]
                    if len(far) == 0:
                        done = True #If no candidates for skeleton particles exist, stop
                        break

            if len(far) > 0:
                finalround = score[far]
                taken.append(far[np.argmax(finalround)])

            if it > 100:
                done = True

        self.skel = np.vstack((self.skel, cluster[taken, :]))

    def plotSkel(self, title='EulerianSkeleton'):
        self.plot = skeletonPlot(self.bound3d, self.skel, folder=self.folder)
        self.plot.snapPlot(title)


class optimumCellSize:
    """Class used to analyze the effect of cell size on the number of populated Eulerian cells"""
    def __init__(self, cluster, labels, eps, k0=0.088, k1=0.3):
        ks = np.linspace(k0, k1)
        filled = np.zeros(len(ks))
        for i,k in enumerate(ks):
            app = eulerianAnalysis(cluster, labels, eps, kx=k, ky=k)
            filled[i] = app.testSnapshot()*100

        plt.figure()
        plt.plot(ks, filled)
        plt.xlabel('Cell Size [-]')
        plt.ylabel('Percentage of Populated Cells [%]')
        plt.title('Evolution of Filled Cells with Cell Size')
