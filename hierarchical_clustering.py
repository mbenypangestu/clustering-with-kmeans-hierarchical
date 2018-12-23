import math
import pandas as pd
import matplotlib.pyplot as plt
import tkinter
import matplotlib.cm as cm
import numpy as np

class CoordinatePoint:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

class Distance:
    def __init__(self, src, dst, distance):
        self._src = src
        self._dst = dst
        self._distance = distance

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, src):
        self._src = src

    @property
    def dst(self):
        return self._dst

    @dst.setter
    def dst(self, dst):
        self._dst = dst

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distance):
        self._distance = distance


class Data:
    def __init__(self, index, x, y):
        self._index = index
        self._x = x
        self._y = y
        self._class_set = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @property
    def class_set(self):
        return self._class_set

    @class_set.setter
    def class_set(self, class_set):
        self._class_set = class_set


class Cluster:
    def __init__(self, index):
        self._index = index
        self._data = []
        self._centroid_point = None
        self._vc2 = None
        self._avg_data = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index   ):
        self._index = index

    @property
    def centroid_point(self):
        return self._centroid_point

    def set_centroid_point(self, centroid_point):
        self._centroid_point = centroid_point

    def setNewClusterCentroidPoint(self):
        data_length = len(self.data)
        temp_x = 0
        temp_y = 0

        for i in range(data_length):
            temp_x =  temp_x + self.data[i].x
        temp_x = temp_x / data_length

        for i in range(data_length):
            temp_y =  temp_y + self.data[i].y
        temp_y = temp_y / data_length

        point = CoordinatePoint(temp_x, temp_y)
        self.set_centroid_point(point)


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def addData(self, data):
        self._data.append(data)

    @property
    def vc2(self):
        return self._vc2

    def setVc2(self, vc2):
        self._vc2 = vc2

    @property
    def avg_data(self):
        return self._avg_data

    def setAvgData(self, avg):
        self._avg_data = avg

class CentroidLinkage:
    def __init__(self):
        self._datas     = []
        self._clusters  = []
        self._distances_data    = []
        self._distances_cluster = []
        self._n = None

    @property
    def datas(self):
        return self._datas

    def addToDatas(self, data):
        self._datas.append(data)

    @property
    def clusters(self):
        return self._clusters

    def addToClusters(self, cluster):
        self._clusters.append(cluster)

    @property
    def distances_data(self):
        return self._distances_data

    def addToDistancesData(self, distance):
        self._distances_data.append(distance)

    @property
    def distances_cluster(self):
        return self._distances_cluster

    def addToDistancesCluster(self, distance):
        self._distances_cluster.append(distance)

    def resetDistanceCluster(self):
        self._distances_cluster.clear()

    def countDistancePerData(self):
        for i in range(len(self.datas)):
            for j in range(len(self.datas)):
                euc_distance  = countEuclidianDistance(abs(self.datas[i].x - self.datas[j].x),
                                                       abs(self.datas[i].y - self.datas[j].y))
                temp_distance = Distance(self.datas[i].index,
                                          self.datas[j].index,
                                          euc_distance)
                self._distances_data.append(temp_distance)

    def start(self, n):
        self._n = n
        self.setPointCluster()

        len_first_cluster = len(self.clusters)
        while len_first_cluster > n:
            self.centroidLinkage()
            self.showClusters()
            len_first_cluster = len_first_cluster - 1

    def centroidLinkage(self):
        print("\n\n=========================||| New Cluster of Centroid Linkage |||==========================\n")
        print("Cluster Length : ", len(self.clusters), " => ", len(self.clusters) - 1)

        self.setDistanceCluster()
        min_distance = self.checkMinDistanceCluster()
        print("Src and Dst data to group : ", min_distance.src, " <=>", min_distance.dst, "\nMin Distance : ", min_distance.distance)

        self.groupingCluster(min_distance)

    def groupingCluster(self, min_distance_cluster):
        new_cluster = Cluster(min_distance_cluster.src)

        idx_min_distance_clust = self.distances_cluster.index(min_distance_cluster)
        self.distances_cluster.pop(idx_min_distance_clust)

        for i in range(len(self.clusters)):
            if self.clusters[i].index == min_distance_cluster.src or self.clusters[i].index == min_distance_cluster.dst:
                for x in range(len(self.clusters[i].data)):
                    new_cluster.addData(self.clusters[i].data[x])

                if self.clusters[i].index == min_distance_cluster.src:
                    idx_src = self.clusters.index(self.clusters[i])
                    print("Src index distance cluster : ", idx_src, end='')
                if self.clusters[i].index == min_distance_cluster.dst:
                    idx_dst = self.clusters.index(self.clusters[i])
                    print(", Dst index distance cluster : ", idx_dst)

        self.clusters.append(new_cluster)
        self.resetDistanceCluster()
        for i in range(len(self.clusters)):
            if self.clusters[i].index == min_distance_cluster.src:
                self.clusters.pop(i)
                break
        for i in range(len(self.clusters)):
            if  self.clusters[i].index == min_distance_cluster.dst:
                self.clusters.pop(i)
                break
        new_cluster.setNewClusterCentroidPoint()


    def setPointCluster(self):
        for i in range(len(self.clusters)):
            self.clusters[i].set_centroid_point( CoordinatePoint(self._datas[i].x, self._datas[i].y) )

    def setDistanceCluster(self):
        for i in range(len(self.clusters)):
            for j in range(len(self.clusters)):
                euc_distance  = countEuclidianDistance( abs(self.clusters[i].centroid_point.x - self.clusters[j].centroid_point.x),
                                                        abs(self.clusters[i].centroid_point.y - self.clusters[j].centroid_point.y))
                temp_distance = Distance(self.clusters[i].index,
                                         self.clusters[j].index,
                                         euc_distance)
                self._distances_cluster.append(temp_distance)

    def checkMinDistanceCluster(self):
        min_distance_range = 9999999999
        distance = None

        for i in range(len(self.distances_cluster)):
            if self.distances_cluster[i].distance != 0:
                if self.distances_cluster[i].distance < min_distance_range:
                    min_distance_range = self.distances_cluster[i].distance
                    distance = self.distances_cluster[i]
        return distance

    def showClusters(self):
        print()
        for i in range(len(self.clusters)):
            print(self.clusters[i].index, "\t=>\t", end='')
            for j in range(len(self.clusters[i].data)):
                print(self.clusters[i].data[j].index, ", ", end='')
            print()
        print("==========================================================================================")

    def showClusterDistance(self):
        print("============== showClusterDistance ==============")
        for i in range(len(self.distances_cluster)):
            print(self.distances_cluster[i].src, " => ", self.distances_cluster[i].dst, " = ",
                  self.distances_cluster[i].distance)
        print("==================================================")

    def showCentroidPoint(self):
        print("============== showCentroidPoint =================")
        for i in range(len(self.clusters)):
            print(self.clusters[i].index, " => ", self.clusters[i].centroid_point.x, ", ",
                  self.clusters[i].centroid_point.y)
        print("==================================================")

class VarianceAnalysis:
    def __init__(self, clusters, datas):
        self._clusters = clusters
        self._datas = datas

    @property
    def clusters(self):
        return self._clusters
    @property
    def datas(self):
        return self._datas

    def analyze(self):
        print("\n============================== Variance Cluster Analysis ==================================")
        self.countVarianceClusters()
        self.countV()
        print("=============================================================================================")

    def countVarianceClusters(self):
        for i in range(len(self.clusters)):
            print(len(self.clusters[i].data))
            sum_d = 0
            for j in range(len(self.clusters[i].data)):
                euc2 = pow(countEuclidianDistance( abs(self.clusters[i].data[j].x - self.clusters[i].centroid_point.x),
                                              abs(self.clusters[i].data[j].y - self.clusters[i].centroid_point.y))
                           , 2)
                sum_d = sum_d + euc2

            vc2 = 1 / (len(self.clusters[i].data) - 1) * sum_d
            self.clusters[i].setVc2(vc2)

    def countV(self):
        N = len(self.datas)
        k = len(self.clusters)

        sum_vc = 0
        for i in range(len(self.clusters)):
            n = len(self.clusters[i].data)
            sum_vc = sum_vc + ((n - 1) * self.clusters[i].vc2)
        vw = 1 / (N - k) * sum_vc

        for i in range(len(self.clusters)):
            n = len(self.clusters[i].data)
            euc2 = pow(countEuclidianDistance(abs(self.clusters[i].centroid_point.x),
                                              abs(self.clusters[i].centroid_point.y))
                       , 2)
            sum_vc = sum_vc + (n * euc2)
        vb = 1 / (k - 1) * sum_vc

        v = (vw / vb)
        print("Error Analysis = ", v)

    def showClusters(self):
        print("============================= Variance Analysis ==========================================")
        for i in range(len(self.clusters)):
            print(self.clusters[i].index, "\t=>\t", end='')
            for j in range(len(self.clusters[i].data)):
                print(self.clusters[i].data[j].index, ", ", end='')
            print()
        print("==========================================================================================")


def main():
    input_file = input("Apa nama file data set ? ")
    input_cluster = input("Berapa jumlah cluster yang diinginkan ? ")

    file = pd.read_csv(input_file, sep=',')

    datafile = pd.DataFrame(file)

    centroidLinkage = CentroidLinkage()

    for i in range(len(datafile)):
        d = Data( i, datafile["A"][i], datafile["B"][i] )
        clust = Cluster(i)

        centroidLinkage.addToDatas(d)
        clust.addData(d)
        centroidLinkage.addToClusters(clust)

    centroidLinkage.countDistancePerData()
    centroidLinkage.start(int(input_cluster))

    varianceAnalysis = VarianceAnalysis(centroidLinkage.clusters, centroidLinkage.datas)
    varianceAnalysis.analyze()

    # centroidLinkage.showCentroidPoint()
    if input_file == "RuspiniDataset.csv":
        showDefaultGraph(centroidLinkage)

def countEuclidianDistance(*params):
    squaredParams = 0
    for param in params:
        squaredParams = squaredParams + math.pow(param, 2)
    return math.sqrt(squaredParams)

def showData(object):
    for i in range(len(object.datas)):
        print(object.datas[i].x, ", ", object.datas[i].y)

def showClusters(object):
    for i in range(len(object.clusters)):
        print(object.clusters[i].index, "\t=>\t", end='')
        for j in range(len(object.clusters[i].data)):
            print(object.clusters[i].data[j].index, ", ", end='')
        print()

def showDistancePerData(object):
    print("============= showDistancePerData ================")
    for i in range(len(object.distances_data)):
        print(object.distances_data[i].src, " => ", object.distances_data[i].dst, " : ", object.distances_data[i].distance)
    print("==================================================")


def showDefaultGraph(object):
    r = 0
    g = 1
    b = 0
    for i in range(len(object.clusters)):
        for j in range(len(object.clusters[i].data)):
            plt.scatter(object.clusters[i].data[j].x, object.clusters[i].data[j].y, color=(r, g, b, 1))
        r = r + 0.2
        g = g - 0.2
        b = b + 0.2

    plt.show()


if __name__ == '__main__':
    main()