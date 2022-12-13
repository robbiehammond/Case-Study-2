# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from scipy.spatial import distance
import math


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

def findClose(points, x, y, time, xdir, ydir, speed, timetable, dists):
    #for other_point in points:
    dps = speed  / 3600000000 # deg per sec

    for t in range(1, 600):
        new_x = x + xdir * dps * t
        new_y = y + ydir * dps * t
        #if no points at this time, give up
        if (float(time + t) not in timetable.keys()):
            continue
        possible_points = timetable[time + t]
        for otherpoint in possible_points:
            dist = distance.euclidean((new_x, new_y), (otherpoint[3], otherpoint[4])) 
            if (dist < .02):
                # if this point has already been assigned, don't reassign it
                if ((otherpoint[3], otherpoint[4]) in dists.keys()):
                    continue
                else:
                    dists[(otherpoint[3], otherpoint[4])] = dist
                    return otherpoint
    return None

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    if numVessels > 10:
        numVessels = 10
    # Unsupervised prediction, so training data is unused
    timetable = {}
    points = []
    dists = {}
    num_rows,_ = testFeatures.shape
    data = np.concatenate((-1*np.ones(num_rows)[:, np.newaxis], testFeatures), axis=1)    # labels
    data = np.concatenate((np.arange(num_rows)[:, np.newaxis], data), axis=1)             # indices
    data = preprocess(data)
    # Create dictionary for faster data transfer
    for i in range(len(data)):
        if data[i][2] not in timetable.keys():
            timetable[data[i][2]] = []
        timetable[float(data[i][2])].append(data[i])
    # initialize list of starting and ending points (for dataset 3)
    start_end_points = [(4569,7321),(2072,6461),(1,1641), (28,8055), (0,7215), (987,4146), (135,8054), (3,4223), (4,8031), (3714,7157)]
    # Classify k clusters based on calculated trajectories
    for k in range(numVessels):
        # initialize start points 
        start_point = data[start_end_points[k][0]]
        end_point = data[start_end_points[k][1]]
        cur_point = start_point
        # follow trajectory as far as possible
        while (cur_point[2] < end_point[2]): 
            mag = data[i][5]
            xdir = data[i][6]
            ydir = data[i][7]
            next_point = findClose(points, cur_point[3], cur_point[4], cur_point[2], xdir, ydir, mag, timetable, dists)
            if (np.array_equal(next_point, cur_point)):
                #print("no next point found, same as previous")
                break
            elif next_point is not None:
                cur_point[1] = k    # set cluster label
                points.append(cur_point)    # add labeled point to list
                cur_point = next_point
            else:
                #print("no next point found")
                break
   
    # loop through any points that are not labeled
    for point in range(len(data)):
        if data[point][0] not in [i[0] for i in points]:
            min_dist = 0.2
            label = 1
            # find closest point in labeled points
            for i in range(len(points)):
                potential_point = points[i]
                dist = distance.euclidean((potential_point[3], potential_point[4]), (data[point][3], data[point][4]))
                if dist < min_dist:
                    min_dist = dist
                    label = potential_point[1]
                if dist < 0.03:
                    min_dist = dist
                    label = potential_point[1]
                    break
            data[point][1] = label
            points.append(data[point])
    points.sort(key=lambda e : e[2])
    predVesselsWithK = np.array([i[1] for i in points])
    return predVesselsWithK

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Assume 10 vessels based on visual inspection
    return predictWithK(testFeatures, 10, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    from utils import loadData, plotVesselTracks
    data = loadData('set3noVID.csv')
    features = data[:,2:]
    labels = data[:,1]

    # Prediction with specified number of vessels
    numVessels = 10
    predVesselsWithK = predictWithK(features, numVessels)


    '''from utils import loadData, plotVesselTracks
    data = loadData('set1.csv')
    features = data[:,2:]
    labels = data[:,1]'''

    #%% Plot all vessel tracks with no coloring

    plotVesselTracks(features[:,[2,1]])
    plt.title('All vessel tracks')
    
    #%% Run prediction algorithms and check accuracy
    
    # Prediction with specified number of vessels
    numVessels = np.unique(labels).size
    predVesselsWithK = predictWithK(features, numVessels)
    ariWithK = adjusted_rand_score(labels, predVesselsWithK)
    
    # Prediction without specified number of vessels
    predVesselsWithoutK = predictWithoutK(features)
    predNumVessels = np.unique(predVesselsWithoutK).size
    ariWithoutK = adjusted_rand_score(labels, predVesselsWithoutK)
    
    print(f'Adjusted Rand index given K = {numVessels}: {ariWithK}')
    print(f'Adjusted Rand index for estimated K = {predNumVessels}: '
          + f'{ariWithoutK}')

    #%% Plot vessel tracks colored by prediction and actual labels
    plotVesselTracks(features[:,[2,1]], np.array(predVesselsWithK))
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], np.array(predVesselsWithoutK))
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    plt.show()

# %%
