# -*- coding: utf-8 -*-
"""
Utility functions for working with AIS data

@author: Kevin S. Xu
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from matplotlib import markers,colors

def convertTimeToSec(timeVec):
    # Convert time from hh:mm:ss string to number of seconds
    return sum([a * b for a, b in zip(
            map(int, timeVec.decode('utf-8').split(':')), [3600, 60, 1])])


def loadData(filename):
    # Load data from CSV file into numPy array, converting times to seconds
    timestampInd = 2

    data = np.loadtxt(filename, delimiter=",", dtype=float, skiprows=1, 
                      converters={timestampInd: convertTimeToSec})
    return data

def plotVesselTracks(latLon, clu=None):
    # Plot vessel tracks using different colors and markers with vessels
    # given by clu
    
    n = latLon.shape[0]
    if clu is None:
        clu = np.ones(n)
    cluUnique = np.array(np.unique(clu), dtype=int)
    
    plt.figure()
    markerList = list(markers.MarkerStyle.markers.keys())
    
    normClu = colors.Normalize(np.min(cluUnique),np.max(cluUnique))
    for iClu in cluUnique:
        objLabel = np.where(clu == iClu)
        imClu = plt.scatter(
                latLon[objLabel,0].ravel(), latLon[objLabel,1].ravel(),
                marker=markerList[iClu % len(markerList)],
                c=clu[objLabel], 
                norm=normClu, 
                label=iClu)
    plt.colorbar(imClu)
    plt.legend().set_draggable(True)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def split_on_VID(data):
    #find all unique VIDs
    s = set()
    for e in data:
        s.add(e[1])

    # map each VID to a list of points, initially empty
    VIDtoPoints = {}
    for vid in s:
        VIDtoPoints[vid] = []

    # populate lists
    for e in data:
        vid = e[1]
        # VIDtoPoints[vid] = [(time1, latitiude1, longitude1, SOG1, COG1), ...]
        VIDtoPoints[vid].append([e[2], e[3], e[4], e[5], e[6], e[7]])
    
    # sort each list of tuples by their first element (i.e. sort them in time)
    for vid in VIDtoPoints:
        sorted(VIDtoPoints[vid], key=lambda e : e[0])
    return VIDtoPoints

def build_KD_tree(data):
    positional_data = np.zeros((len(data), 2))
    for i in range(len(data)):
        longitude = data[i][2]
        latitiude = data[i][3]
        positional_data[i] = np.array([longitude, latitiude])
    return (positional_data, KDTree(positional_data))

def closest_point_map(data):
    positional_data, tree = build_KD_tree(data)
    
    closestPoints, inds = tree.query(positional_data, k = 2) #first point is just the same point
    return (closestPoints[:,1], inds[:,1])
