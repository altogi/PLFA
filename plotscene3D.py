import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D
from temporalTracking import temporalTrackerGlobal

class plotscene3D:
    def __init__(self, times, cluster=1, angles=[[45, 45], [90, -45]], domain=np.array([[0.108, 0.008, 0.013], [0.162, 0.032, 0.028]]), folder=''):
        """For the inputs:
        times: Time instants to follow, in 100 iterations (100 its = 0.15 ms)
        cluster: Initial cluster label to follow
        angles: List of the form [[azim1, elev1], [azim2, elev2]...] with as many pairs of angles as subscene per time instant
        domain: Bounds of spatial domain"""
        self.times = times
        self.angles = angles
        self.domain = domain
        self.folder = folder
        self.L = np.linalg.norm(domain[1, :] - domain[0, :])

        if len(times) > 1:
            pairs = []
            for t1, t2 in zip(times, times[1:]):
                pairs.append([t1, t2])
            self.tt = temporalTrackerGlobal(pairs, plot=False)
            _, _, _, self.sequence = self.tt.trackCluster(cluster, plot=False)
        else:
            self.sequence = [cluster]

        self.fig = plt.figure()
        self.fig.set_size_inches(18.5, 11)
        self.axs = []
        for i in range(len(angles)):
            ax = self.fig.add_subplot(1, len(angles), i + 1, projection=Axes3D.name)
            self.axs.append(ax)

    def isolateCluster(self, folder='./Data/v78/t_'):
        self.data = []
        for i, s in enumerate(self.sequence):
            root = folder + str(self.times[i])
            labels = np.load(root + '/ClusterLabels.npy')
            data = np.load(root + '/originalZ_cut.npy')

            mask = labels == s
            self.data.append(data[mask])

    def snapPlot(self, title='Evolution of Cluster '):
        def update(choose):
            self.plotInstance(choose)
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            for a in self.axs:
                a.cla()
            return image

        imageio.mimsave(self.folder + title + str(self.sequence[0]) + '.gif', [update(i) for i in range(len(self.times))], fps=4)

    def plotInstance(self, i):
        t = self.times[i] * 0.15 / 100
        self.this = self.data[i]
        self.fig.suptitle('t [ms] = ' + '{:.2f}'.format(t), fontsize=26)
        for i, a in enumerate(self.angles):
            self.singleScene(a[0], a[1], self.axs[i])

    def singleScene(self, azim, elev, ax):
        ax.set_xlabel('\nx [m]', fontsize=18)
        ax.set_ylabel('\ny [m]', fontsize=18)
        ax.set_zlabel('\nz [m]', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.view_init(elev, azim)

        pov = self.L * np.array([np.cos(elev* np.pi / 180) * np.cos(azim* np.pi / 180), np.cos(elev* np.pi / 180) * np.sin(azim* np.pi / 180), np.sin(elev* np.pi / 180)])
        c = np.zeros(len(self.this))
        for i, r in enumerate(self.this):
            c[i] = np.linalg.norm(r - pov)
        c = (c - np.min(c)) / (np.max(c) - np.min(c))

        ax.scatter(self.this[:, 0], self.this[:, 1], self.this[:, 2], c=c, cmap='inferno', alpha=0.5, s=0.5)
        # tit = 'Elev = ' + str(elev) + '. Azim = ' + str(azim)
        # ax.set_title(tit, fontsize=22)

        plt.tight_layout()

        # ax.set_xlim3d(self.domain[0, 0], self.domain[1, 0])
        # ax.set_ylim3d(self.domain[0, 1], self.domain[1, 1])
        # ax.set_zlim3d(self.domain[0, 2], self.domain[1, 2])
        X = self.domain[:, 0]
        Y = self.domain[:, 1]
        Z = self.domain[:, 2]
        # From https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

for i in range(3, 10):
    p = plotscene3D(times = [i for i in range(0, 4200, 100)], cluster=i)
    p.isolateCluster()
    p.snapPlot()