import math
import random as rand
import pandas as pd
import matplotlib.pyplot as plt
import tkinter
import matplotlib.cm as cm
import numpy as np

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
    def __init__(self, index, *coordinates):
        self._index = index
        self._coordinates = []
        self._class_set = None
        self._nearest_centroid = None
        self.addCoordinates(coordinates)

    @property
    def index(self):
        return self._index

    def setIndex(self, index):
        self._index = index

    @property
    def coordinates(self):
        return self._coordinates

    def addCoordinate(self, coordinate):
        self._coordinates.append(coordinate)

    def addCoordinates(self, coordinates):
        for coord in coordinates:
            self.coordinates.append(coord)

    @property
    def class_set(self):
        return self._class_set

    def setClass_set(self, class_set):
        self._class_set = class_set

    @property
    def nearest_centroid(self):
        return self._nearest_centroid

    def setNearest_centroid(self, nearest_centroid):
        self._nearest_centroid = nearest_centroid


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

class KMeans :
    def __init__(self, n_cluster):
        self._datas     = []
        self._clusters  = []
        self._distances_data    = []
        self._distances_cluster = []
        self._n_cluster = int(n_cluster)
        self._centroid_points = []
        self._new_centroid_points = []
        self._distances_to_centroid = []

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

    @property
    def centroid_points(self):
        return self._centroid_points

    def addCentroidPoint(self, centroid):
        self.centroid_points.append(centroid)

    @property
    def new_centroid_points(self):
        return self._new_centroid_points

    def addNewCentroidPoint(self, centroid):
        self.new_centroid_points.append(centroid)

    @property
    def distances_to_centroid(self):
        return self._distances_data

    def addDistancesToCentroid(self, distance):
        self._distances_to_centroid.append(distance)

    def start(self):
        self.generateRandomCentroid()

        while(True):
            self.countDistanceDataToCentroid()
            self.setNearestCentroid()
            self.setNewCentroidPosition()

            if (self.isNewCentroidSameLastCentroid() == False):
                self.centroid_points.clear()

                for i in range(len(self.new_centroid_points)):
                    self.centroid_points.append(self.new_centroid_points[i])
                print(False)
            else:
                print(True)
                break

        self.setClusters()

    def generateRandomCentroid(self):
        for i in range(0, self._n_cluster):
            rand_centroid = Data(i, rand.randrange(1, 200, 3), rand.randrange(1, 200, 3))
            print(rand.randrange(1, 200, 3), ", ", rand.randrange(1, 200, 3))
            self.centroid_points.append(rand_centroid)

    def countDistanceDataToCentroid(self):
        print("\n========================Distance data to centroid=======================================")
        for i in range(len(self.datas)):
            for j in range(len(self.centroid_points)):
                euc_distance  = countEuclidianDistance(abs(self.datas[i].coordinates[0] - self.centroid_points[j].coordinates[0]),
                                                       abs(self.datas[i].coordinates[1] - self.centroid_points[j].coordinates[1]))
                temp_distance = Distance(self.datas[i].index,
                                          self.centroid_points[j].index,
                                          euc_distance)
                print(self.datas[i].index, ", ",
                      self.centroid_points[j].index, " = ",
                      euc_distance)
                self.distances_to_centroid.append(temp_distance)

    def setNearestCentroid(self):
        print("\n========================Nearest centroid=======================================")
        for i in range(len(self.datas)):
            temp_distance_nearest_centroid = 999999999.99
            temp_index_nearest_centroid = None

            for j in range(len(self.distances_to_centroid)):
                if self.distances_to_centroid[j].src == i :
                    if self.distances_to_centroid[j].distance < temp_distance_nearest_centroid:
                        temp_distance_nearest_centroid = self.distances_to_centroid[j].distance
                        temp_index_nearest_centroid = self.distances_to_centroid[j].dst

            self.datas[i].setNearest_centroid(temp_index_nearest_centroid)
            print(self.datas[i].index, " = ", self.datas[i].nearest_centroid)

    def setNewCentroidPosition(self):
        print("\n========================New centroid point=======================================")
        new_centroid_points = []
        for i in range(len(self.centroid_points)):
            temp_x_total = 0
            temp_y_total = 0
            len_centroid_choosen = 0

            for j in range(len(self.datas)):
                if (self.datas[j].nearest_centroid == self.centroid_points[i].index):
                    print(self.datas[j].nearest_centroid, " <==> ", self.centroid_points[i].index)
                    temp_x_total = temp_x_total + self.datas[j].coordinates[0]
                    temp_y_total = temp_y_total + self.datas[j].coordinates[1]
                    len_centroid_choosen = len_centroid_choosen + 1


            if len_centroid_choosen != 0:
                new_centroid_i = Data(i, (temp_x_total / len_centroid_choosen), (temp_y_total / len_centroid_choosen))
            else:
                new_centroid_i = Data(i, self.centroid_points[i].coordinates[0], self.centroid_points[i].coordinates[1])

            new_centroid_points.append(new_centroid_i)

        self.new_centroid_points.clear()
        for i in range(len(new_centroid_points)):
            print(new_centroid_points[i].coordinates[0], ", ", new_centroid_points[i].coordinates[1])
            self.new_centroid_points.append(new_centroid_points[i])

    def isNewCentroidSameLastCentroid(self):
        result = True

        for i in range(len(self.new_centroid_points)):
            if (self.new_centroid_points[i].coordinates[0] != self.centroid_points[i].coordinates[0] or
                self.new_centroid_points[i].coordinates[1] != self.centroid_points[i].coordinates[1]):
                result = False
                return result

        return result

    def setClusters(self):
        for i in range(len(self.centroid_points)):
            clust = Cluster(i)

            for j in range(len(self.datas)):
                if self.datas[j].nearest_centroid == i:
                    clust.addData(self.datas[j])

            self.clusters.append(clust)



def main():
    input_file = input("Apa nama file data set ? ")
    input_cluster = input("Berapa jumlah cluster yang diinginkan ? ")

    file = pd.read_csv("RuspiniDataset.csv", sep=',')

    datafile = pd.DataFrame(file)

    kmean = KMeans(input_cluster)

    for i in range(len(datafile)):
        d = Data( i, datafile["A"][i], datafile["B"][i] )
        kmean.addToDatas(d)

    # showData(kmean)
    kmean.start()
    showClusters(kmean)
    showDefaultGraph(kmean)

def countEuclidianDistance(*params):
    squaredParams = 0
    for param in params:
        squaredParams = squaredParams + math.pow(param, 2)
    return math.sqrt(squaredParams)

def showData(object):
    print("================ Data ==================")
    for i in range(len(object.datas)):
        for j in range(len(object.datas[i].coordinates)):
            print(object.datas[i].coordinates[j], " , ", end='')
        print()
    print("========================================")

def showClusters(object):
    print("\n====================================== Clusters ===========================================")
    for i in range(len(object.clusters)):
        print(object.clusters[i].index, "\t=>\t", end='')
        for j in range(len(object.clusters[i].data)):
            print(object.clusters[i].data[j].index, ", ", end='')
        print()
    print("===========================================================================================")

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
            plt.scatter(object.clusters[i].data[j].coordinates[0], object.clusters[i].data[j].coordinates[1], color=(r, g, b, 1))
        r = r + 0.2
        g = g - 0.2
        b = b + 0.2

    plt.show()


if __name__ == '__main__':
    main()

