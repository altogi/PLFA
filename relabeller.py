import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from clusterPlot import clusterPlot
import time

# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: Based on a previous DBSCAN cluster labeling for every 2D snapshot in Z, and a set of labeled skeleton
# particles, which takes into account their connectivity across different 2D snapshots in Z, this script is in charge of
# replacing the old DBSCAN label assigned to every cluster particle by the label of the skeleton particles within each cluster.
# For every skeleton particle, the closest cluster particle is determined, and its DBSCAN label is extracted. Then, all
# cluster particles sharing such DBSCAN label have their cluster label set to the label of the skeleton particle. This script is
# also in charge of determining the area in each 2D snapshot in Z which belongs to each of the new skeleton particle labels.
# Álvaro Tomás Gil - UIUC 2020

class relabeller:
	"""Based on a previous DBSCAN cluster labeling for every 2D snapshot in Z, and a set of labeled skeleton
	particles, which takes into account their connectivity across different 2D snapshots in Z, this script is in charge of
	replacing the old DBSCAN label assigned to every cluster particle by the label of the skeleton particles within each cluster.
	For every skeleton particle, the closest cluster particle is determined, and its DBSCAN label is extracted. Then, all
	cluster particles sharing such DBSCAN label have their cluster label set to the label of the skeleton particle. This script is
	also in charge of determining the area in each 2D snapshot in Z which belongs to each of the new skeleton particle labels. Inputs:
	data: 2D array containing all particle locations projected into a set of 2D planes for different values of Z
	oldlab: Array containing the DBSCAN labels assigned to each particle of data. As many elements as data. Label -1 corresponds to non-cluster particles
	skel: 2D array containing the positions of all skeleton particles describing the cluster's interior
	newlab: Array of labels assigned to every skeleton particles. As many elements as skel.
	oldAreas: List of dicts containing as many elements as Z levels. Every dict has, for a DBSCAN label as a key, the assined area as a value
	folder: folder in which to save resulting plots"""
	def __init__(self, data, oldlab, skel, newlab, oldAreas, folder=''):
		self.data = data
		self.skel = skel
		self.newlab = newlab
		self.oldAreas = oldAreas
		self.folder = folder

		self.zs = np.unique(data[:, 2])
		self.dz = np.abs(self.zs[1] - self.zs[0])/4
		self.dZ = np.abs(self.zs[1] - self.zs[0])
		self.clusters = data[oldlab != -1]
		self.oldlab = oldlab[oldlab != -1]
		self.nextL = max(self.newlab)

		self.new = np.zeros(len(self.oldlab)) #0 for _ in range(len(self.oldlab))]
		self.areCluster = np.ndarray.flatten(np.argwhere(oldlab != -1))
		self.nbrs = NearestNeighbors(n_neighbors = 1, algorithm = 'auto').fit(self.clusters)

	def run(self):
		print('Relabelling of Cluster Particles')
		start = time.time()
		for i, z in enumerate(self.zs):
			print('	Z = ' + str(round(z, 4)) + ' - Percentage Relabelled: ' + str(round(len(np.nonzero(self.new)[0])*100/len(self.new), 3)) + '%.')
			self.forZsnap(z)

		self.verify()

		self.newLab = np.zeros(len(self.data))
		for i, c in enumerate(self.areCluster):
			self.newLab[c] = self.new[i]

		print('Execution Time: ' + str(round(time.time() - start, 3)))
		self.labelPlot()
		self.computeVolume()


	def forZsnap(self, thisZ):
		"""Method in charge of isolating cluster and skeleton particles relevant to the current 2D snapshot in Z and executing the procedure"""
		#relevantS indexes skel with skeleton particles within the Z level described by thisZ
		self.relevantS = np.arange(len(self.skel))[((self.skel[:, 2] >= thisZ - self.dz) & (self.skel[:, 2] <= thisZ + self.dz))]
		self.spneigh = NearestNeighbors(n_neighbors=15, algorithm='auto').fit(self.skel[self.relevantS])
		# relevantC indexes clusters with cluster particles within the Z level described by thisZ
		self.relevantC = np.arange(len(self.clusters))[((self.clusters[:, 2] >= thisZ - self.dz) & (self.clusters[:, 2] <= thisZ + self.dz))]
		for i in self.relevantS:
			self.forSkelPoint(i)
		self.assignZeros()


	def forSkelPoint(self, i):
		"""Method in charge of extracting a skeleton particle, its label, its closest cluster particle, and the label of the latter. Then,
		the method calls updateLabels"""
		sp = self.skel[i, :]
		closest = self.nbrs.kneighbors(np.reshape(sp, (1, -1)), return_distance = False)
		loserL = self.oldlab[closest[0][0]]
		winnerL = self.newlab[i]

		self.updateLabels(winnerL, loserL)

	def updateLabels(self, winner, loser):
		"""Method in charge of examining all cluster particles with label 'loser' and  assigning label 'winner' to them"""
		take = [(self.new[i] == 0 and self.oldlab[i] == loser) for i in self.relevantC]
		intersect = [(self.new[i] != 0 and self.oldlab[i] == loser and self.new[i] != winner) for i in self.relevantC]
		toChange = self.relevantC[take] #these cluster particles havent had their labels substituted by one of an SP before
		toJudge = self.relevantC[intersect] #these have, but such label is different than the one currently imposed

		for i in toChange:
			self.new[i] = winner

		for i in toJudge: #What if a cluster particle has already had its label updated?
			self.new[i] = self.decideIntersect(winner, self.new[i], i)


	def decideIntersect(self, l1, l2, i):
		"""This method is in charge of resolving a conflict in new label assignment. If a cluster particle is to be relabelled
		with a skeleton particle label when it has already been assigned another skeleton particle label, this cluster particle
		is to be assigned the skeleton particle label of the skeleton particle which is closest of the two."""
		r = self.clusters[i, :]
		SP1 = [j for j in self.relevantS if self.newlab[j] == l1]
		SP2 = [j for j in self.relevantS if self.newlab[j] == l2]
		dist, closest = self.spneigh.kneighbors(np.reshape(r, (1, -1)))
		# closest indexes relevantS, not skel!
		# relevantS, and thus SP1 and SP2 do index skel
		winner = 0
		for sp, d in enumerate(dist[0]):
			if self.relevantS[closest[0][sp]] in SP1:
				winner = l1
			elif self.relevantS[closest[0][sp]] in SP2:
				winner = l2

			if winner > 0:
				break

		if winner == 0:
			print('	Intersection between SP labels ' + str(l1) + ' and ' + str(l2) + ' yielded no result.')
			sp = self.relevantS[closest[0][sp]]
			winner = self.newlab[sp]
			print('	Winner is then SP label ' + str(winner))
		return winner

	def assignZeros(self):
		"""Since it may be the case that smaller clusters are not assigned a skeleton particle, this method is in charge of
		generating a new skeleton particle label exclusively for these clusters."""
		abandoned = [i for i in self.relevantC if self.new[i] == 0]
		labels = np.unique([self.oldlab[i] for i in abandoned])
		for l in labels:
			self.nextL += 1
			assign = [i for i in abandoned if self.oldlab[i] == l]
			for j in assign:
				self.new[j] = self.nextL

	def verify(self):
		"""Once the relabeling procedure is finished, this method is in charge of verifying if all skeleton particle labels
		coincide with the labels of the closest cluster particles."""
		self.agree = [False for i in range(len(self.skel))]
		for i,sp in enumerate(self.skel):
			closest = self.nbrs.kneighbors(np.reshape(sp, (1, -1)), return_distance = False)
			if self.new[closest[0][0]] == self.newlab[i]:
				self.agree[i] = True
			else:
				print('	SP ' + str(i) + ' at ' + str(sp) + ' with label ' + str(self.newlab[i]) + ' has a cluster label of ' + str(self.new[closest[0][0]]))

		print('	From the ' + str(len(self.skel)) + ' skeleton particles present, ' + str(sum(self.agree)) + ' (' + str(round(sum(self.agree)*100/len(self.skel), 3)) + '%) coincide with their relabelled cluster label.')


	def labelPlot(self):
		"""Plotting of relabelled clusters."""
		self.plot = clusterPlot(self.clusters, self.new, self.folder)
		self.plot.plotAll('Relabelled Clusters')

	def computeVolume(self):
		"""Based on previously areas per new label, the algorithm estimates the cluster's volume based on a linear interpolation."""
		self.createIndex()
		self.volumes = [0 for _ in np.unique(self.new)]

		for i, z in enumerate(self.zs[:-1]):
			areas1 = self.newAreas[i + 1]
			areas0 = self.newAreas[i]
			for k in areas0.keys():
				if k in areas1.keys():
					self.volumes[k - 1] += 0.5 * (areas0[k] + areas1[k]) * self.dZ
				else:
					self.volumes[k - 1] += 0.5 * areas0[k] * self.dZ

			for k in areas1.keys():
				if k not in areas0.keys():
					self.volumes[k - 1] += 0.5 * areas1[k] * self.dZ

		self.volumePlot()

	def createIndex(self):
		"""For every initial DBSCAN label, this method determines to which new skeleton particle label this label has been converted.
		This method is also in charge of comparing old and new labelling of cluster particles, and based on the areas per 2D snapshot in Z
		assigned to each old cluster label, compute the areas per 2D snapshot assigned to each new cluster label. """
		self.new = self.new.astype(int)
		self.newAreas = [0 for _ in self.zs]
		for i,z in enumerate(self.zs):
			oldAreas = self.oldAreas[i]
			areadict = {}
			self.relevantC = np.arange(len(self.clusters))[
				((self.clusters[:, 2] >= z - self.dz) & (self.clusters[:, 2] <= z + self.dz))]
			maxNew = np.max(self.new[self.relevantC])
			maxOld = np.max(self.oldlab[self.relevantC])
			index = np.zeros((maxOld + 1, maxNew))
			for j, l1, l2 in zip(self.relevantC, self.oldlab[self.relevantC], self.new[self.relevantC]):
				index[l1, l2 - 1] += 1

			#For every old label, divide each of the transitions to each skeleton particle label by the total in the old label
			for j, row in enumerate(index):
				if np.sum(row) == 0:
					print('e')
				index[j] = row/np.sum(row)

			#For every new label, compute the associated area based on the old estimated areas
			for j, col in enumerate(np.transpose(index)):
				total = 0
				for k, fraction in enumerate(col):
					if k in oldAreas.keys():
						total += oldAreas[k]*fraction

				areadict[j + 1] = total

			self.newAreas[i] = areadict

	def volumePlot(self, top=10):
		"""This method is simply in charge of plotting a bar plot comparing cluster volumes"""
		fig = plt.figure()
		fig.set_size_inches(18.5, 9.5)
		ax = fig.add_subplot(111)
		label = ['Cluster ' + str(i) for i in range(1, len(self.volumes) + 1)]

		top = min(top, len(self.volumes))
		volumesC = np.sort(self.volumes)[::-1][:top]
		sortI = np.argsort(self.volumes)[::-1][:top]
		label = [label[i] for i in sortI]

		cmap = plt.get_cmap('plasma')
		c = cmap(volumesC)

		ax.bar(range(top), volumesC, tick_label=label, width=0.5, color=c)
		ax.tick_params(labelsize=18)
		plt.ylabel('Volume [m^3]', fontsize=18)
		plt.title('Volume per Cluster', fontsize=22)
		plt.savefig(self.folder + 'Volume per Cluster')
		plt.close('all')

