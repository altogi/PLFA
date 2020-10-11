import numpy as np
import matplotlib.pyplot as plt
from clustering3D import clustering3D
from clusterConnect import skeletonConnect
from relabeller import relabeller
from eulerianApproach import eulerianAnalysis, eulerianSkeleton, optimumCellSize
from boundaryFinder import boundaryFinder
import time

# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script carries out all the steps of analysis for a single data frame of the simulation of
# particle-laden turbulent flow. Parting from a .csv datafile of particle positions sorted into a set of dicrete planes
# of constant Z, this script applies DBSCAN clustering to each of these snapshots in Z to separate cluster from void
# particles, obtains a set of closed curves defining each clusters' boundaries, generates a constellation of interior
# points defining each cluster's brute skeleton, connects these interior points along different levels of Z, and based
# on these connections assigns a set of cluster labels which associate each cluster particle to a three-dimensional cluster.
# Álvaro Tomás Gil - UIUC 2020


class clusteringAnalysis:
    """This class carries out all the steps of analysis for a single data frame of the simulation of
    particle-laden turbulent flow. Parting from a .csv datafile of particle positions sorted into a set of dicrete planes
    of constant Z, this script applies DBSCAN clustering to each of these snapshots in Z to separate cluster from void
    particles, obtains a set of closed curves defining each clusters' boundaries, generates a constellation of interior
    points defining each cluster's brute skeleton, connects these interior points along different levels of Z, and based
    on these connections assigns a set of cluster labels which associate each cluster particle to a three-dimensional cluster.
    For the inputs:
    filename: This is the name of the .csv file from which to begin this analysis. Note that with different time frames,
    projection sheet thicknesses, or spacings between these sheets, parts of this filename will vary
    t: This number references the current time frame to analyze. Based on this time frame, a different folder will be treated"""
    def __init__(self, filename='prt_TG_ductVe8_770000_Large_dZ_0,01_Spacing_0,02', t=0, root='./Data/t_'):
        self.t = t
        self.folder = root + str(t) + '/'
        self.filename = self.folder + filename + '.csv'
        self.times = [0, 0, 0, 0, 0, 0]

    def run(self, skip=0, time=False):
        """This is the main method of the class.
        skip: Int which sets at which point of the analysis to stop loading previous data and start generating new data.
        time: Boolean setting whether to plot the time consumption of different steps"""
        self.clustering(loadNskip=skip > 0)
        self.boundaries(loadNskip=skip > 1)
        self.eulerize(loadNskip=skip > 2)
        self.skeletonize(loadNskip=skip > 3)
        self.connect(loadNskip=skip > 4)
        self.relabel()

        if time:
            self.timePiePlot()

    def clustering(self, loadNskip=False, take=20, knownMinPts=210, knownEps=0.0015953):
        """This method is in charge of applying DBSCAN clustering analysis to separate the particles contained in every
        separate plane of constant Z into cluster and void particles. Special care goes into the definition of the
        parameters with which to carry out DBSCAN, both requiring different methods of class clustering3D to be invoked.
        For the inputs:
        loadNskip: Boolean which commands the method to skip execution and directly return previously saved results
        take: Number of planes of constant Z to analyze from the original .csv dataset
        knownMinPts: Since the determination of an un-biased value of this parameter is costly, one can directly input
        an adequate value instead.
        knownEps: Since the determination of an un-biased value of this parameter is costly, one can directly input
        an adequate value instead.
        For the outputs:
        MinPts: MinPts parameter with which DBSCAN is carried out
        eps: epsilon parameter with which DBSCAN is carried out
        data: 2D array of 3D particle positions, sorted into parallel planes of constant Z
        DBSCANlabels: list of equal lenght as data, assigning a cluster label applicable within each 2D domain. A label
        of -1 corresponds to a void particle."""
        start = time.time()
        if loadNskip:
            self.MinPts = np.load(self.folder + 'MinPts.npy')
            self.eps = np.load(self.folder + 'eps.npy')
            self.data = np.load(self.folder + 'allPoints.npy')
            self.DBSCANlabels = np.load(self.folder + 'DBSCANlabels.npy')
        else:
            # 3D DBSCAN clustering of Particle Coordinates
            self.c3d = clustering3D(self.filename, take=take, folder=self.folder)
            # Extraction of Unbiased Hyperparameters
            self.c3d.MinPts(known=knownMinPts)
            self.c3d.meanReach(known=knownEps)
            self.c3d.sweep()

            self.MinPts = self.c3d.MinPts
            self.eps = self.c3d.eps
            self.data = self.c3d.data
            self.DBSCANlabels = self.c3d.labels

            np.save(self.folder + 'MinPts.npy', self.c3d.MinPts)
            np.save(self.folder + 'eps.npy', self.c3d.eps)
            np.save(self.folder + 'allPoints.npy', self.c3d.data)
            np.save(self.folder + 'DBSCANlabels.npy', self.c3d.labels)
        self.times[0] = time.time() - start

    def boundaries(self, loadNskip=False):
        """This second method has the task of extracting the cluster particles in each cluster which serve as boundaries
        with respect to the rest of the domain.
        For the input:
        loadNskip: Boolean which commands the method to skip execution and directly return previously saved results
        For the outputs:
        bound3d: 2D array of 3D positions corresponding to the curves defining each cluster's boundaries for each of the
        parallel planes of constant Z in the domain
        """
        start = time.time()
        if loadNskip:
            self.bound3d = np.load(self.folder + 'bound3d.npy')
        else:
            self.bF = boundaryFinder(self.data, self.DBSCANlabels, self.eps, folder=self.folder)
            self.bF.sweep()
            self.bound3d = self.bF.bound3d

            np.save(self.folder + 'bound3d.npy', self.bF.bound3d)
        self.times[1] = time.time() - start

    def eulerize(self, loadNskip=False):
        """This method discretizes each plane of constant Z by means of a uniform grid. This discretization is used to
        compute the area associated to each two-dimensional DBSCAN cluster as well as to generate the skeleton inner to
        the cluster.
        For the input:
        loadNskip: Boolean which commands the method to skip execution and directly return previously saved results
        For the outputs:
        DBSCANareas: list of dicts, describing the area associated to each DBSCAN cluster label in a single Z level. This
        list has as many elements as different parallel planes of constant Z exists, and each of these elements is a dict
        whose keys are the DBSCAN labels of the Z level and whose values are the areas of those clusters.
        cellCenters: 2D array of 3D positions corresponding to the cell centers of all discretized planes of constant Z
        cellLabels: 1D array of equal length as cellCenters, where each element corresponds to the DBSCAN label assigned to
        each grid cell, based on the DBSCAN label of the particles it contains."""
        start = time.time()
        if loadNskip:
            self.DBSCANareas = np.load(self.folder + 'DBSCANareas.npy', allow_pickle=True)
            self.cellCenters = np.load(self.folder + 'cellCenters.npy')
            self.cellLabels = np.load(self.folder + 'cellLabels.npy')
        else:
            self.eu = eulerianAnalysis(self.data, self.DBSCANlabels, self.eps, boundMethod=None, folder=self.folder)
            self.eu.run()

            self.DBSCANareas = self.eu.areas
            self.cellCenters = self.eu.rc
            self.cellLabels = self.eu.lc

            np.save(self.folder + 'DBSCANareas.npy', self.eu.areas)
            np.save(self.folder + 'cellCenters.npy', self.eu.rc)
            np.save(self.folder + 'cellLabels.npy', self.eu.lc)
        self.times[2] = time.time() - start

    def skeletonize(self, loadNskip=False):
        """Taking into account the previous discretization of each two-dimensional domain, this method determines a set
        of points interior to the two-dimensional domain which condense its topology.
        For the input:
        loadNskip: Boolean which commands the method to skip execution and directly return previously saved results
        For the outputs:
        skel: 2D array of 3D positions of these skeleton points
        """
        start = time.time()
        if loadNskip:
            self.skel = np.load(self.folder + 'skeletonize.npy')
        else:
            self.sk = eulerianSkeleton(self.cellCenters, self.cellLabels, self.bound3d, folder=self.folder)
            self.sk.run()

            self.skel = self.sk.skel

            np.save(self.folder + 'skeletonize.npy', self.sk.skel)
        self.times[3] = time.time() - start

    def connect(self, loadNskip=False):
        """Once these interior points have been generated, this method examines the connectivity between them, both for
        the same level of Z as well as along different levels. Based on these connectivities, the method groups the
        skeleton points into different cluster labels, which take into account the 3D connectivities within different
        clusters at different planes of constant Z.
        For the input:
        loadNskip: Boolean which commands the method to skip execution and directly return previously saved results
        For the outputs:
        SKlabels: List of equal length as skel, assigning a cluster label to each skeleton point.
        """
        start = time.time()
        if loadNskip:
            self.SKlabels = np.load(self.folder + 'SKlabels.npy')
        else:
            self.skC = skeletonConnect(self.skel, self.bound3d, folder=self.folder)
            self.skC.run()

            self.SKlabels = self.skC.labels

            np.save(self.folder + 'SKlabels.npy', self.skC.labels)
        self.times[4] = time.time() - start

    def relabel(self):
        """Lastly, this method applies the cluster labels obtained by examining connectivities of different skeleton points
        to the cluster particles grouped by means of DBSCAN. This method results in a new list of labels to apply to each
        cluster particles, taking into account cluster connections in 3D."""
        start = time.time()
        self.reL = relabeller(self.data, self.DBSCANlabels, self.skel, self.SKlabels, self.DBSCANareas, self.folder)
        self.reL.run()
        self.times[5] = time.time() - start
        np.save(self.folder + 'DBSCANVolumes.npy', self.reL.volumes)
        np.save(self.folder + 'ClusterLabels.npy', self.reL.newLab)

    def timePiePlot(self, pctM=0.00):
        """This method simply generates a plot of the time consumption associated to each step of the analysis."""
        names = ['clustering', 'boundary', 'discretize', 'skeletonize', 'connectivities', 'relabel']
        dict = {}
        for i,j in zip(names, self.times):
            dict[i] = j

        total = sum(dict.values())
        title = 'Time Consumption per Step of Analysis  - Total [s]= ' + str(round(total, 3))
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
        plt.pie(dict.values(), labels=labels, shadow=True, startangle=0, labeldistance=labdis, colors=colors, textprops={'fontsize': 14})
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


# root = ['prt_TG_ductVe8_78', '00_Large_dZ_0,01_Spacing_0,02']
# # root = ['prt_TG_ductVe8_77', '00_Large_dZ_0,01_Spacing_0,02']
# folder = './Data/v78/t_'
# times = [i for i in range(0, 42)] + [65, 90]
# for t in times:
#     t0 = str(t)
#     if len(t0) < 2:
#         ts = '0' * (2 - len(t0)) + t0
#     else:
#         ts = t0
#     filename = root[0] + ts + root[1]
#     c = clusteringAnalysis(filename=filename, t=t * 100, root=folder)
#     c.run()