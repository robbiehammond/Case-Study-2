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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

def findClose(points, x, y, time, xdir, ydir, speed, timetable):
    #for other_point in points:
    dps = speed  / 36000 # deg per sec

    for t in range(1, 600):
        new_x = x + xdir * dps * t
        new_y = y + ydir * dps * t
        #if no points at this time, give up
        if (float(time + t) not in timetable.keys()):
            continue
        possible_points = timetable[time + t]
        for otherpoint in possible_points:
           # if points:
               # if otherpoint[0] not in points:   # check if unique ID is already in labeled points
                    dist = distance.euclidean((new_x, new_y), (otherpoint[3], otherpoint[4])) 
                    if (dist < .05):
                        return otherpoint
               # else:
                  #  dist = distance.euclidean((new_x, new_y), (otherpoint[3], otherpoint[4])) 
                  #  old_distance = points.index(otherpoint[0])
                  #  if (dist < points[old_distance][1]):
                   #     return otherpoint, dist
                #print(f"match found at t={t}: start: ({x}, {y}), predicted: ({new_x}, {new_y}), acutal: ({otherpoint[3]}, {otherpoint[4]}), dist: {dist}")
    return None

def predictWithK(testFeatures, numVessels, trainFeatures=None, 
                 trainLabels=None):
    predVessels = []
    # Unsupervised prediction, so training data is unused
    timetable = {}
    points = []
    num_rows,_ = testFeatures.shape
    data = np.concatenate((np.zeros(num_rows)[:, np.newaxis], testFeatures), axis=1)    # labels
    data = np.concatenate((np.arange(num_rows)[:, np.newaxis], data), axis=1)           # indices
    print(data.shape)
    data = preprocess(data)
    # Create dictionary for faster data transfer
    for i in range(len(data)):
            if data[i][2] not in timetable.keys():
                timetable[data[i][2]] = []
            timetable[float(data[i][2])].append(data[i])
    # initialize list of starting and ending points (for dataset 3)
    start_end_points = [(4569,7321),(2072,6461),(1,1641), (28,8055), (0,7215), (987,4146), (135,8054), (3,4223), (4,8031), (3714,7157)]
    #import random
    #random.seed(2)
    #random.shuffle(start_end_points)
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
            next_point = findClose(points, cur_point[3], cur_point[4], cur_point[2], xdir, ydir, mag, timetable)
            if (np.array_equal(next_point, cur_point)):
                print("no next point found, same as previous")
                break
            elif next_point is not None:
                cur_point[1] = k    # set cluster label
                points.append(cur_point)    # add labeled point to list
                cur_point = next_point
            else:
                print("no next point found")
                break
        #print("time at end: ", cur_point[2])
        #print("lat/long at end: ", cur_point[3], cur_point[4])
        #print("terminated")
    
    # TODO: add remaining points to nearest cluster 
    return points

def predictWithoutK(testFeatures, trainFeatures=None, trainLabels=None):
    # Unsupervised prediction, so training data is unused
    
    # Assume 10 vessels based on visual inspection
    return predictWithK(testFeatures, 10, trainFeatures, trainLabels)

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    from utils import loadData, plotVesselTracks
    data = loadData('set3noVID.csv')
    features = data[:,2:]
    # labels = data[:,1]

    # Prediction with specified number of vessels
    numVessels = 10
    predVesselsWithK = predictWithK(features, numVessels)
    #ariWithK = adjusted_rand_score(labels, predVesselsWithK)


    '''from utils import loadData, plotVesselTracks
    data = loadData('set1.csv')
    features = data[:,2:]
    labels = data[:,1]'''

#%%
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    points_np = np.array(predVesselsWithK)
    ax.scatter3D(points_np[:,3], points_np[:,4], points_np[:,2], c=points_np[:,1])
    ax.set_ylabel('Longitude')
    ax.set_xlabel('Latitude')
    ax.set_zlabel('Time')
    ax.legend()
    plt.show()

    #%% Plot all vessel tracks with no coloring
    '''
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
    plotVesselTracks(features[:,[2,1]], predVesselsWithK)
    plt.title('Vessel tracks by cluster with K')
    plotVesselTracks(features[:,[2,1]], predVesselsWithoutK)
    plt.title('Vessel tracks by cluster without K')
    plotVesselTracks(features[:,[2,1]], labels)
    plt.title('Vessel tracks by label')
    plt.show()'''
