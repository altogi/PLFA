import numpy as np
import vtk
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt


# Development of data analysis tools for the topological and temporal analysis of clusters of particles in turbulent flow
# Script Description: This script is designed to open a specified VTK file representing the positions of particles in a
# simulation of particle-laden flow and extract their position for a thin plane bisecting the simulation's domain. Class
# multipleZs is in charge of invoking class VTKopen, in order to define a series of computational frames out a dataset
# describing turbulent particle-laden flow within a square duct.
# Álvaro Tomás Gil - UIUC 2020

class VTKopen:
    """Class in charge of reading the actual .vtk file, cropping and projecting its positions based on the specified ranges,
    and saving the results to a .csv file.
    For the inputs:
    filename: String with file name
    xrange & yrange: Lists of two elements defining ranges in X and Y, defining the span of each computational plane,
    normalized w.r.t the duct dimensions in each direction
    zs: Thickness of the sheet associated to every computational frame, defining the amount of particles to project onto
    the plane. This thickness is normalized w.r.t. the duct dimension in Z
    dZ: Spacing between adjacent computational planes, normalized w.r.t. the duct dimension in Z
    normalize: Boolean. If True, normalized coordinates will appear"""
    def __init__(self, filename, xrange, yrange, zs, dZ, normalize=False):
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename + '.vtk')
        reader.Update()
        self.polydata = reader.GetOutput()

        if np.isscalar(zs):
            zs = np.array([zs])

        self.r = np.array([0, 0, 0])
        self.fullr = [np.array([0, 0, 0])] * len(zs)
        self.originalZ = [np.array([0, 0, 0])] * len(zs)
        self.x = xrange
        self.y = yrange
        self.zs = zs
        self.dZ = dZ
        self.n = self.polydata.GetNumberOfPoints()
        self.norm = normalize

        #Initial sweep of the data file, to determine dimensions in each direction for plotting and normalization.
        for i in range(0, self.n, 1000):
            self.r = np.vstack((self.r, list(self.polydata.GetPoint(i))))
        self.r = self.r[1:]

        self.dims = np.zeros(3)
        for i in range(3):
            self.dims[i] = round(max(self.r[:, i]), 2)
            self.r[:, i] = self.r[:, i] / max(self.r[:, i])

    def read(self, filename='test'):
        """This method is in charge of reading every line of the .vtk file, extracting positions within the specified
        range, normalizing if necessary, and saving the extracted data as a .csv file"""

        total = 0
        for i in range(0, self.n, 1):
            vraw = list(self.polydata.GetPoint(i))
            v = [vraw[j] / self.dims[j] for j in range(3)]

            if v[0] > self.x[0] and v[0] < self.x[1] and v[1] > self.y[0] and v[1] < self.y[1]:
                which = np.vstack((v[2] < self.zs + self.dZ, v[2] > self.zs - self.dZ))
                which = np.ndarray.flatten(np.argwhere(np.all(which, axis=0)))
                # "which" describes within which sheet (or Z level) such a point should be classified
                if len(which) > 0:
                    total += 1
                    if self.norm:
                        original = v.copy()
                        v[2] = self.zs[which[0]]
                        temp = np.vstack((self.fullr[which[0]], v))
                    else:
                        original = vraw.copy()
                        vraw[2] = self.zs[which[0]] * self.dims[2]
                        temp = np.vstack((self.fullr[which[0]], vraw))

                    self.fullr[which[0]] = temp
                    temp2 = np.vstack((self.originalZ[which[0]], original))
                    self.originalZ[which[0]] = temp2

            print(float(100 * i / self.n), '% percent read')

        for i in range(len(self.zs)):
            self.fullr[i] = self.fullr[i][1:, :]
            self.originalZ[i] = self.originalZ[i][1:, :]

        print(str(total) + ' particles in data set')

        self.data = np.array([0, 0, 0])
        self.dataZ = np.array([0, 0, 0])
        for i,r in enumerate(self.fullr):
            self.data = np.vstack((self.data, r))
            self.dataZ = np.vstack((self.dataZ, self.originalZ[i]))
        self.data = self.data[1:, :]
        self.dataZ = self.dataZ[1:, :]

        np.savetxt(filename + '.csv', self.data, delimiter=",")
        cutt = filename.split('/')
        newf = ''
        for c in cutt[:-1]:
            newf = newf + c + '/'
        np.save(newf + 'originalZ', self.dataZ)

    def plots(self, onlyHist=True):
        """This method is in charge of developing a histogram of particle concentration along Y, to study the influence of the
        duct walls, and of plotting the particle positions for a single Z value"""
        plt.figure()
        plt.hist(self.r[:, 1]*self.dims[1], bins=100)
        plt.xlabel('y [m]')
        plt.ylabel('Number of Particles [-]')
        plt.title('Concentration of Particles along Y')

        if not onlyHist:
            plt.figure()
            plt.scatter(self.fullr[:, 0], self.fullr[:, 1], s=0.5)
            plt.title(str(len(self.fullr)) + ' particles in data set')
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.axis('equal')
        plt.show()


class multipleZs:
    """Class in charge of invoking VTKopen, by introducing the desired values of Z for each computational plane"""
    def __init__(self, filename='prt_TG_ductVe8_780000', xrange=[0.4, 0.6], yrange=[0.2, 0.8], thick=0.01, spacing=0.02, wallMargin=0.2, normalize=False):
        self.dZ = thick
        self.sp = spacing
        self.margin = wallMargin
        self.x = xrange
        self.y = yrange
        self.filename = filename
        self.norm = normalize

        self.zs = np.unique(
            np.concatenate((np.arange(0.5, 1 - self.margin, self.sp), np.arange(self.margin, 0.5, self.sp))))
        self.vtk = VTKopen(filename, xrange, yrange, self.zs, self.dZ, normalize)

    def writeFile(self):
        thick = str(round(self.dZ, 3)).replace('.', ',')
        spacing = str(round(self.sp, 3)).replace('.', ',')
        filename = self.filename + '_Large_dZ_' + thick + '_Spacing_' + spacing

        self.vtk.read(filename)

#root = ['./Data/t_','/prt_TG_ductVe8_77']
root = ['./Data/v78/t_','/prt_TG_ductVe8_78']
times = [65, 90]# + [i for i in range(0, 42)]
for t in times:
    t0 = str(t * 100)
    if len(t0) < 4:
        ts = '0' * (4 - len(t0)) + t0
    else:
        ts = t0
    filename0 = root[0] + t0 + root[1] + ts
    vt = multipleZs(filename0)
    vt.writeFile()


