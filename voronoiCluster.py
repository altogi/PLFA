import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import time
from voronoiPlot import voronoiPlot

# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is designed perform the 3D particle cluster analysis of a dataset of particle positions
# employing Voronoi tesselations. First, the dataset of positions is cropped to a specified set of dimensions. Then,
# the Voronoi cells corresponding to the particles are determined, and those of a volume corresponding to a cluster particle
# will be isolated. Then the connectivity of these Voronoi cells will be studied, in order to determine global cluster volumes.
# Lastly, this script allows for a topological validation of a set of skeleton points, by examining what percentage of these
# skeleton points exist in regions which this Voronoi analysis classifies as belonging to particle clusters.
# Álvaro Tomás Gil - UIUC 2020

class VoronoiCluster:
    """This is the main class of the script, in charge of performing the 3D particle cluster analysis of a dataset of particle positions
    employing Voronoi tesselations. First, the VTK dataset of positions is cropped to a specified set of dimensions. Then,
    the Voronoi cells corresponding to the particles are determined, and those of a volume corresponding to a cluster particle
    will be isolated. Then the connectivity of these Voronoi cells will be studied, in order to determine global cluster volumes. For the input:
    data: 2D array of 3D positions of all particles
    filename: If data is None, VTK file from which to load the data
    ranges: Limits of the domain in the form [[xmin, xmax], [ymin, ymax], [zmin, zmax]]"""
    def __init__(self, ranges=[[0.108, 0.162], [0.008, 0.032], [0.013, 0.028]], data=None, filename='prt_TG_ductVe8_780000.vtk', folder='./Data/v78/t_0/'):
        print('Clustering Analysis of Turbulent Particle-laden Flow with Voronoi Volumes')
        self.ranges = np.array(ranges)
        self.data = data
        self.times = [0, 0]

        if data is None:
            self.loadVTK(filename, folder)


    def loadVTK(self, filename, folder):
        """This method is in charge of loading the VTK file in order to obtain an un-projected set of particle positions"""
        import vtk
        print('Extracting Dataset')
        start = time.time()
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(folder + filename)
        reader.Update()
        polydata = reader.GetOutput()
        n = polydata.GetNumberOfPoints()
        self.data = np.array([0, 0, 0])

        for i in range(0, n, 1):
            vraw = list(polydata.GetPoint(i))
            inRange = np.all([vraw[0] > self.ranges[0,0], vraw[0] < self.ranges[0,1], vraw[1] > self.ranges[1,0], vraw[1] < self.ranges[1,1], vraw[2] > self.ranges[2,0], vraw[2] < self.ranges[2,1]])
            if inRange:
                self.data = np.vstack((self.data, np.array(vraw)))
                if i % 50000 == 0:
                    print(' Out of the ' + str(n) + ' particles in the dataset, ' + str(i) + ' (' + str(round(i*100/n, 3)) + ' %) have been processed, and ' + str(len(self.data) - 1) + ' have been stored.')

        self.data = self.data[1:, :]
        rangeStr = '_x[' + str(self.ranges[0,0]) + ',' + str(self.ranges[0,1]) + ']_y[' + str(self.ranges[1,0]) + ',' + str(self.ranges[1,1]) + ']_z[' + str(self.ranges[1,0]) + ',' + str(self.ranges[1,1]) + '].npy'
        np.save(folder + 'VoronoiData' + rangeStr, self.data)
        print('Elapsed Time: ' + str(round(time.time() - start, 3)))

    def volumePDF(self, maxVar=-1, bins=75, threshold=1):
        """This method is in charge of carrying out the Voronoi Tessellation of the supplied data, and obtaining a PDF
        of the resulting normalized Voronoi volumes. It also compares this PDF with the one that would result from
        a set of Poisson distributed points."""
        print('Cluster Identification Based on Voronoi Volumes')
        start = time.time()
        self.vor = Voronoi(self.data)
        self.volumes, self.nonB = self.voronoiVolumes(self.vor)
        self.nonBI = np.arange(0, len(self.vor.point_region))[self.nonB]
        self.volumes_sorted = np.sort(self.volumes)
        self.oldOrder = np.argsort(self.volumes)

        if maxVar > 0:
            means = [np.mean(self.volumes_sorted)]
            varMean = []
            topV = -1
            #Discard some very big Voronoi cells which unnecessarily alter the mean volume. Stop once the mean volume does
            #not vary more than maxVar with an elimination of these large cells. Deactivate this part with maxVar= < 0
            for i in range(250):
                volumes = self.volumes_sorted[:-(i + 1)]
                means.append(np.mean(volumes))
                varM = (means[-1] - means[-2])/means[-2]
                varMean.append(varM)
                if np.abs(varM) < maxVar and topV == -1:
                    topV = -(i + 1)
            self.oldOrder = self.oldOrder[:topV]
            self.volumes_sorted = self.volumes_sorted[:topV]

        self.V = self.volumes_sorted/np.mean(self.volumes_sorted)
        self.bins = np.logspace(np.log(np.min(self.V)), np.log(np.max(self.V)), bins)

        self.PDF, _ = np.histogram(self.V, bins=self.bins, density=True)
        self.bins = (self.bins[1:] + self.bins[:-1]) / 2

        self.RandomPDF = self.PoissonPDF(self.bins)
        self.intersectPDFs(threshold=threshold)
        self.assignLabels()
        self.times[0] = time.time() - start
        print('Elapsed Time: ' + str(round(time.time() - start, 3)))


    def voronoiVolumes(self, vor):
        """Given a Voronoi Object, this method is in charge of obtaining the volume of each Voronoi Cell, and classifying it
        as a boundary cell if one of its vertex indices is -1 or if any of its vertices is outside the domain of interest"""
        volumes = np.array([])
        data = vor.points
        limits = [[np.min(data[:, 0]), np.max(data[:, 0])], [np.min(data[:, 1]), np.max(data[:, 1])], [np.min(data[:, 2]), np.max(data[:, 2])]]
        nonB = [False for _ in data]
        for i, region in enumerate(vor.point_region):
            indices = vor.regions[region]
            if -1 not in indices:
                v = vor.vertices[indices]
                isWithin = self.checkVertices(v, limits)
                if isWithin:
                    volumes = np.append(volumes, ConvexHull(v).volume)
                    nonB[i] = True
        return volumes, nonB

    @staticmethod
    def checkVertices(vertices, limits):
        """Given a set of Voronoi Vertices, this simple methods checks if all of them are maintained within the range
        of the form [[xmin, xmax], [ymin, ymax], [zmin, zmax]] expressed in limits"""
        isWithin = True
        for i,v in enumerate(vertices):
            x = v[0]
            y = v[1]
            z = v[2]
            if x < limits[0][0] or x > limits[0][1]:
                isWithin = False
                break
            if y < limits[1][0] or y > limits[1][1]:
                isWithin = False
                break
            if z < limits[2][0] or z > limits[2][1]:
                isWithin = False
                break
        return isWithin

    @staticmethod
    def PoissonPDF(v):
        """Given a set of normalized Voronoi volumes, this method computes the corresponding PDF, as per Ferenc et al. 1992"""
        from scipy.special import gamma

        a = 3.24174
        b = 3.24269
        c = 1.26861
        g = gamma(a / c)
        k1 = c * b ** (a / c) / g
        pdf = k1 * np.power(v, (a - 1)) * np.exp(- b * np.power(v, c))
        return pdf

    def intersectPDFs(self, threshold=1):
        """This method determines at which normalized Voronoi volumes do the Random PDF and the obtained PDF intersect"""
        diff = np.abs(self.PDF - self.RandomPDF)
        half = np.argmax(self.RandomPDF)
        start = np.nonzero(self.PDF > 0.5*np.max(self.PDF))[0][0]
        end = np.nonzero(self.RandomPDF[half:] < 0.5*np.max(self.RandomPDF))[0][0] + half

        if start == 0 and half == 0:
            self.cut1 = 0
        else:
            self.cut1 = np.argmin(diff[start:half]) + start
        self.V1 = self.bins[self.cut1] * threshold

        self.cut2 = np.argmin(diff[half:end]) + half
        self.V2 = self.bins[self.cut2]

    def assignLabels(self):
        """This obtains a list of indexes of points which can be labeled as cluster particles."""
        clusters = np.arange(0, len(self.V))[self.V < self.V1] #indexes self.V, volumes_sorted, and oldOrder
        self.clusterV = self.volumes_sorted[clusters]
        clusters = self.oldOrder[clusters] #indexes volumes
        self.clusters = self.nonBI[clusters] #indexes self.vor and self.data
        self.easyLabel = np.zeros(len(self.data))
        self.easyLabel[self.clusters] = 1
        print('Out of ' + str(len(self.data)) + ' particles, ' + str(len(self.clusters)) + ' (' + str(round(len(self.clusters)*100/len(self.data), 3)) +' %) are labelled as cluster particles.')

    def optimumBins(self, b0=100, b1=10000, n=100):
        """This method tracks the evolution of the first intersection between PDFs with the number of bins in the PDF"""
        self.intersections = []
        for i in np.linspace(b0, b1, n):
            self.volumePDF(bins=i)
            self.intersections.append(self.V1)

        plt.figure()
        plt.plot(np.linspace(b0, b1, n), self.intersections)
        plt.xlabel('Number of Bins [-]')
        plt.ylabel('Normed Voronoi Volume of Intersection [-]')
        plt.title('Evolution of Intersection Volume with Number of Bins')

    def connectClusterCells(self):
        print('Connectivities of Cluster Cells')
        start = time.time()
        self.connect = connectClusters(self.vor, self.clusters)
        self.connect.run()

        self.labels = self.connect.labels
        self.unique_labels = np.unique(self.labels)
        clusters = [c for c in self.unique_labels if c != -1]
        self.volumesC = [0 for _ in clusters]
        clustLabels = [self.labels[i] for i in self.clusters]
        for i,l in enumerate(clustLabels):
            self.volumesC[l] += self.clusterV[i]

        self.volumePlot()
        self.times[1] = time.time() - start
        print('Elapsed Time: ' + str(round(time.time() - start, 3)))



    def plotVolumePDFs(self, topV=3, noSecond=True):
        """This method plots the obtained PDF and the PDF of the randomly distributed case together with their intersection
        points."""
        take = self.bins < topV
        fig = plt.figure()
        plt.plot(self.bins[take], self.PDF[take], label='Preferential Distribution')
        plt.plot(self.bins[take], self.RandomPDF[take], label='Random Distribution')
        plt.plot(self.V1*np.ones(50), np.linspace(0, self.PDF[self.cut1]), '--', label='First Intersection - V = ' + str(round(self.V1, 2)))
        if not noSecond:
            plt.plot(self.V2 * np.ones(50), np.linspace(0, self.PDF[self.cut2]), '--', label='Second Intersection - V = ' + str(round(self.V2, 2)))
        plt.xlim([0, topV])
        plt.title('Voronoi Cell Volume PDF')
        plt.xlabel('Normed Volume [-]')
        plt.ylabel('PDF [-]')
        plt.legend()

    def plotClusters(self):
        """Plots all particles, sorting them into cluster or non-cluster particles according to the Voronoi classification"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.set_size_inches(18.5, 9.5)
        ax.set_title('Identification of Cluster Particles with Voronoi Volumes', fontsize=22)
        ax.set_xlabel('x [m]', fontsize=18)
        ax.set_ylabel('y [m]', fontsize=18)
        ax.set_zlabel('z [m]', fontsize=18)

        strength = np.linspace(0, 0.8, len(self.unique_labels))
        np.random.shuffle(strength)
        colors = [plt.cm.nipy_spectral(each) for each in strength]
        np.random.shuffle(strength)
        colorsB = [plt.cm.nipy_spectral(each) for each in strength]

        for k, col, colB in zip(self.unique_labels, colors, colorsB):
            a = 1
            s = 3
            if k == -1:
                # Black used for noise.
                col = [1, 0, 0]
                a = 0.3
                s = 1

            class_member_mask = (self.labels == k)
            xy = self.data[class_member_mask]
            if len(xy) > 0:
                ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=np.reshape(np.array(col), (1, -1)),
                           edgecolors=np.reshape(np.array(colB), (1, -1)), alpha=a, s=s, label='Cluster ' + str(k))


    def plotVolumeContours(self):
        """Plots all particles, coloring them as a function of their associated Voronoi cell volume"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.set_size_inches(18.5, 9.5)
        ax.set_title('Particle Positions Colored by Voronoi Volumes', fontsize=22)
        ax.set_xlabel('x [m]', fontsize=18)
        ax.set_ylabel('y [m]', fontsize=18)
        ax.set_zlabel('z [m]', fontsize=18)
        pos = ax.scatter(self.data[self.nonB, 0], self.data[self.nonB, 1], self.data[self.nonB, 2], s=10, c=self.volumes, cmap='plasma')
        cbar = fig.colorbar(pos, ax=ax)
        cbar.ax.tick_params(labelsize=15)

    def plotVoronoiCell(self, cells):
        """Plots a single Voronoi cell, with its Voronoi vertices as well. To gain perspective wrt to the rest of the points,
        the limits of the plots are set according to the limits of all the point positions."""
        for i in cells:
            #i indexes volumes
            i = self.nonBI[i] #now i indexes vor.point_region

            vI = self.vor.regions[self.vor.point_region[i]]
            v = self.vor.vertices[vI, :]
            r = v

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            fig.set_size_inches(18.5, 9.5)
            ax.set_title('Voronoi Cell of Particle ' + str(i))
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_zlabel('z [m]')
            ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=5, alpha=0.5, label='Cell Boundaries')
            ax.scatter(self.data[i, 0], self.data[i, 1], self.data[i, 2], s=25, label='Cell Center')
            ax.set_xlim3d(np.min(self.data[:, 0]), np.max(self.data[:, 0]))
            ax.set_ylim3d(np.min(self.data[:, 1]), np.max(self.data[:, 1]))
            ax.set_zlim3d(np.min(self.data[:, 2]), np.max(self.data[:, 2]))
            # limits = np.vstack((np.array([np.max(self.data[:, 0]), np.max(self.data[:, 1]), np.max(self.data[:, 2])]), np.array([np.min(self.data[:, 0]), np.min(self.data[:, 1]), np.min(self.data[:, 2])])))
            # ax.scatter(limits[:, 0], limits[:, 1], limits[:, 2], s=1)
            ax.legend()

    def volumePlot(self, top=10):
        """This method is simply in charge of plotting a bar plot comparing cluster volumes"""
        fig = plt.figure()
        fig.set_size_inches(18.5, 9.5)
        ax = fig.add_subplot(111)
        label = ['Cluster ' + str(i) for i in range(1, len(self.volumesC) + 1)]

        volumesC = np.sort(self.volumesC)[::-1][:top]
        sortI = np.argsort(self.volumesC)[::-1][:top]
        label = [label[i] for i in sortI]

        cmap = plt.get_cmap('plasma')
        c = cmap(volumesC)

        ax.bar(range(top), volumesC, tick_label=label, width=0.5, color=c)
        ax.tick_params(labelsize=18)
        plt.ylabel('Volume [m^3]', fontsize=18)
        plt.title('Volume per Cluster', fontsize=22)
        plt.savefig('Voronoi Volumes per Cluster')

    def timePiePlot(self, pctM=0.04):
        """This method simply generates a plot of the time consumption associated to each step of the analysis."""
        names = ['Voronoi Tesselation', 'Connectivity of Cluster Cells']
        dict = {}
        for i,j in zip(names, self.times):
            dict[i] = j

        total = sum(dict.values())
        title = 'Time Consumption per Step of Voronoi Analysis  - Total [s]= ' + str(round(total, 3))
        labels = []
        values = []
        for v in dict.keys():
            if dict[v] / total > pctM:
                labels.append(v + ' - ' + str(round(dict[v], 3)) + ' (' + str(round(dict[v] * 100 / total, 2)) + ' %)')
            else:
                labels.append(v)
            values.append(dict[v] / total)
        labdis = 1.07
        cmap = plt.get_cmap("plasma")
        c = np.arange(len(dict.keys())) / len(dict.keys())
        colors = cmap(c)

        fig = plt.figure()
        fig.set_size_inches(12, 7)
        plt.title(title, fontsize=22)
        plt.pie(dict.values(), labels=labels, shadow=True, startangle=0, labeldistance=labdis, colors=colors)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


class connectClusters:
    """This class connects Voronoi cells labeled as cluster cells into larger clusters, by applying a cluster label to
    each Voronoi cell. For the input:
    vor: Voronoi object from scipy.spatial of a dataset of particle positions
    clusters: List of indices indexing the dataset employed in vor, referencing non-boundary Voronoi cells which are cluster cells"""
    def __init__(self, vor, clusters):
        self.vor = vor
        self.clusters = clusters.astype(int)

        self.N = len(clusters)
        self.isCluster = [i in self.clusters for i in range(len(self.vor.point_region))]
        self.labels = [-1 for _ in self.vor.point_region]
        self.maxLabel = 0
        self.pairs = vor.ridge_points
        self.taken = []
        self.it = 0

    def run(self):
        """Main method of the class"""
        for i,p in enumerate(self.pairs):
            self.forPointPair(i)
            if i % 100000 == 0:
                print('Percentage Processed: ' + str(round(i * 100 / len(self.pairs), 3)) + '. Existing Cluster Labels: ', len(np.unique(self.labels)))

    def forPointPair(self, i):
        """For a Voronoi ridge specified by i, this method processes the adjacent Voronoi cell centers, assigning
        the corresponding cluster label to each of them."""
        areCluster = [self.isCluster[j] for j in self.pairs[i]]

        if sum(areCluster) > 1:
            #If at least two neighboring cells are cluster cells, four possible cases exist: 1. none of them have been previously
            #labeled and thus a new cluster label has to be defined, 2. all have been labeled with the same cluster label
            #and as a result nothing is to be done, 3. only few of them has been labeled with a cluster label which is
            #then propagated to the other cells, 4. or several have been assigned different cluster labels, and thus the older
            #cluster label has to be propagated.

            labels = [self.labels[j] for j in self.pairs[i]]
            already = [j != -1 for j in labels]
            if sum(already) == 0: #None of the cell centers have been assigned a cluster label
                for j,p in enumerate(self.pairs[i]):
                    if areCluster[j]:
                        self.labels[p] = self.maxLabel
                self.maxLabel += 1
            else: #At least one of the cell centers has been assigned a cluster label
                contesting = [j for j in labels if j != -1]
                toAssign = min(contesting)
                for j,p in enumerate(self.pairs[i]):
                    if areCluster[j]:
                        if labels[j] == -1:
                            self.labels[p] = toAssign
                        elif labels[j] != toAssign:
                            self.propagateLabel(toAssign, labels[j])
                self.maxLabel = np.max(self.labels) + 1

    def propagateLabel(self, l1, l2):
        """This method solves a conflict of labels by propagating the older (lower) label to the Voronoi cells labeled with the
        newer label"""

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

            # print('Loser Label is ' + str(loser) + ' . With ' + str(loserN) + ' associated cells. Winner label is ' + str(winner))

class  VoronoiValidation:
    """This class carries out the validation of a set of skeleton points to which a cluster label has been assigned,
    by comparing these labels to the ones which result of performing a 3D clustering analysis with Voronoi. For each
    skeleton point, this class extracts its closest Voronoi cell, as well as the cluster label assigned to such cell. Then,
    for each skeleton cluster label, one can examine what percentage of its skeleton points has been misclassified.
    For the inputs:
    data: 2D array of 3D positions of Voronoi cell centers, or essentially particle positions
    vorLabels: List of same length as data, assigning a Voronoi cluster label to each cell center
    skel: 2D array of 3D positions of skeleton points
    skelLabels: List of same length as skel, assigning a cluster label to each skeleton point
    expelExtreme: boolean determining whether to expel skeleton particles from the upper and lower levels of Z from the analysis"""
    def __init__(self, data, vorLabels, skel, skelLabels, expelExtremes=True):
        self.data = data
        self.vorLabels = [v + 1 for v in vorLabels]
        self.skel = skel
        self.skelLabels = [s - 1 for s in skelLabels]

        if expelExtremes:
            maxZ = np.max(self.skel[:, 2])
            minZ = np.min(self.skel[:, 2])
            expel = [i for i in range(len(self.skel)) if self.skel[i, 2] == maxZ or self.skel[i, 2] == minZ]
            self.skel = np.delete(self.skel, expel, axis=0)
            self.skelLabels = np.delete(self.skelLabels, expel, axis=0)

        self.nbrs = NearestNeighbors(n_neighbors=1).fit(self.data)
        self.uniqueVor = np.unique(self.vorLabels)
        self.uniqueSkel = np.unique(self.skelLabels)
        self.memberships = np.zeros((len(self.uniqueSkel), len(self.uniqueVor)))
        self.isCorrect = [1 for _ in self.skel]

    def run(self):
        """Main method of the class, in charge of examining skeleton cluster label and presenting results"""
        for l in self.uniqueSkel:
            mask = np.arange(len(self.skel))[self.skelLabels == l]
            counts = self.findNearest(mask)
            self.memberships[l] = counts

        #self.memberships is an array of as many rows as skeleton labels and as many columns as Voronoi cluster labels,
        #where the i-th row shows for all skeleton points of cluster label i, how many belong to each of the Voronoi
        #cluster labels. More precisely, the j-th column of the i-th row of this array shows how many skeleton points
        #of cluster label i have a closest Voronoi cell center of label j.

        print('Out of ' + str(len(self.skel)) + ' skeleton points, ' + str(sum(self.memberships[:, 0])) + ' (' + str(round(sum(self.memberships[:, 0]) * 100/len(self.skel), 3)) +  ' %) appear in areas classified as void areas by Voronoi')

        for l in self.uniqueSkel:
            members = sum(self.skelLabels == l)
            topVor = np.argsort(self.memberships[l])[::-1][:5] - 1
            counts = np.sort(self.memberships[l])[::-1][:5]
            print('For the ' + str(members) + ' skeleton points with label ' + str(l) + ': ')
            for i in range(5):
                if counts[i] > 0:
                    if topVor[i] == -1:
                        add = ' ' + str(counts[i]) + ' ( ' + str(round(counts[i] * 100 / members, 3)) + ' %) are not associated with a Voronoi cluster cell'
                    else:
                        add = ' ' + str(counts[i]) + ' ( ' + str(round(counts[i] * 100/ members, 3)) + ' %) belong to the Voronoi Cluster with label ' + str(topVor[i])
                    print(add)

        self.plotResults()

    def findNearest(self, i):
        """For a list i of indexes of skeleton point positions, this method examines the closest Voronoi cel center to each
        skeleton point, and based on this counts how many of the skeleton points belong to each Voronoi label. Note that
        memberships is a vector where its i-th element shows how many of the skeleton positions have a closest Voronoi
        cell of label i."""
        skel = self.skel[i, :]
        closest = self.nbrs.kneighbors(skel, return_distance=False)
        memberships = np.zeros(len(self.uniqueVor))
        for j, c in enumerate(closest):
            c = c[0]
            nearLabel = self.vorLabels[c]
            memberships[nearLabel] += 1
            if nearLabel == 0:
                self.isCorrect[i[j]] = 0
        return memberships

    def plotResults(self):
        """This method plots the skeleton particles labeled according to whether their closest Voronoi cell is classified
        as a cluster cell or not."""

        clusters = self.data[[i for i in range(len(self.data)) if self.vorLabels[i] != 0], :]
        vorLabels = [self.vorLabels[i] for i in range(len(self.data)) if self.vorLabels[i] != 0]

        self.plot = voronoiPlot(clusters, self.skel, self.skelLabels, self.isCorrect, vorLabels)
        self.plot.snapPlot()

# data = np.load('VoronoiValidation_Data.npy')
# tr = VoronoiCluster(data)
# tr.volumePDF()
# np.save('VoronoiValidation_Clusters.npy', tr.clusters)
# np.save('VoronoiValidation_ClusterVolumes.npy', tr.clusterV)
# tr.plotVolumePDFs()
#
# tr.connectClusterCells()
# tr.plotClusters()
# tr.timePiePlot()
# np.save('VoronoiValidation_ClusterLabels.npy', tr.connect.labels)
# np.save('VoronoiValidation_VolumeperCluster.npy', tr.volumesC)
#
# skel = np.load('skeletonize.npy')
# skelLabels = np.load('SKlabels.npy')
# vv = VoronoiValidation(data, tr.connect.labels, skel, skelLabels)
# vv.run()


