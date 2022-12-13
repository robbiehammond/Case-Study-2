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
from scipy.spatial import distance


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


def trigify(deg):
    realDeg = math.radians(int(deg / 10))
    
    return (np.cos(realDeg), np.sin(realDeg))

def preprocess(data):
    COG = 6
    COGs = data[:, COG]
    cosines = []
    sines = []
    for deg in COGs:
        cos, sin = trigify(deg)
        sines.append(sin)
        cosines.append(cos)

    # replace COG with cos, add sin col
    data[:, 6] = cosines
    sines = np.array(sines).reshape(-1, 1)
    data = np.hstack((data, sines))


    return data

def findClose(points, x, y, time, xdir, ydir, speed, timetable):
    #for other_point in points:
    dps = speed  / 36000 # deg per sec

    for t in range(1, 60):
        new_x = x + xdir * dps * t
        new_y = y + ydir * dps * t
        #if no points at this time, give up
        if (float(time + t) not in timetable.keys()):
            continue
        possible_points = timetable[time + t]
        for otherpoint in possible_points:
            dist = distance.euclidean((new_x, new_y), (otherpoint[3], otherpoint[4])) 
            if (dist < .02):
                return otherpoint
                #print(f"match found at t={t}: start: ({x}, {y}), predicted: ({new_x}, {new_y}), acutal: ({otherpoint[3]}, {otherpoint[4]}), dist: {dist}")
    return None