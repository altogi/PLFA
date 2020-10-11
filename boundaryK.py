import numpy as np
import matplotlib.pyplot as plt
from boundaryFinder import boundaryFinder
from sklearn.neighbors import NearestNeighbors


root = ['prt_TG_ductVe8_78', '00_Large_dZ_0,01_Spacing_0,02']
folder = './Data/v78/t_0/'
K = np.linspace(0.15, 1.5, 150)
D = np.array([90, 75, 60, 45, 30])
N = np.zeros((len(K), len(D)))
sep = np.zeros((len(K), len(D)))
MinPts = np.load(folder + 'MinPts.npy')
eps = np.load(folder + 'eps.npy')
data = np.load(folder + 'allPoints.npy')
DBSCANlabels = np.load(folder + 'DBSCANlabels.npy')
z = data[0, 2]
take = data[:, 2] == z
data = data[take]
DBSCANlabels = DBSCANlabels[take]
clusNbrs = NearestNeighbors(n_neighbors=1).fit(data)
aveSep = np.mean(clusNbrs.kneighbors()[0]) / eps

for i, k in enumerate(K):
    for j, d in enumerate(D):
        ang = d * np.pi / 180
        dirs = 2 * np.pi / ang
        b = boundaryFinder(data, DBSCANlabels, eps)
        b.sweep(k=k, dirs=int(dirs), plot3d=False)
        N[i, j] = len(b.bound3d) / sum(take)
        boundNbrs = NearestNeighbors(n_neighbors=1).fit(b.bound3d)
        sep[i, j] = np.mean(boundNbrs.kneighbors()[0]) / eps

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(16, 10)
ax.set_xlabel('k [-]', fontsize=24)
ax.grid(True)
ax.set_ylabel('Boundary Particles / Cluster Particles [-]', fontsize=20)
ax.tick_params(axis='both', labelsize=18)
ax.set_title('Sensitivity of Boundary Criterion with k', fontsize=24)
fig.tight_layout()
for i, d in enumerate(D):
    ax.plot(K, N[:, i], label=r'$\delta =$ ' + str(d) + ' º')
ax.legend(fontsize=20)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(16, 10)
ax.set_xlabel('k [-]', fontsize=24)
ax.grid(True)
ax.set_ylabel(r'Average Separation between Adjacent Particles / $\epsilon$ [-]', fontsize=20)
ax.tick_params(axis='both', labelsize=18)
ax.set_title('Evolution of Average Separation between Adjacent Particles with k', fontsize=24)
fig.tight_layout()
ax.plot(K, aveSep * np.ones(len(K)), label='Average Separation between Cluster Particles')
for i, d in enumerate(D):
    ax.plot(K, sep[:, i], label='Average Separation between Boundary Particles - ' + r'$\delta =$ ' + str(d) + ' º')

ax.legend(fontsize=18)
plt.show()