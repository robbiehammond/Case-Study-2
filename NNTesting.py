# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""

import random
from torch import nn
from scipy.spatial import distance
from sklearn.neighbors import KDTree
from utils import split_on_VID, closest_point_map
import torch
import numpy as np
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

'''
def findClose(f, next_points):
    lat = f[0]
    long = f[1]
    myPos = (lat, long)
    bestPoint = None
    bestInd = None
    minDist = 10000000 #change to float max
    for point in next_points:
        otherPos = (point[0][0], point[0][1])
        d = distance.euclidean(myPos, otherPos)
        # if this point is close to the predicted, lets go with it
        if (d < .1 and d <= minDist):
            minDist = d 
            bestPoint = otherPos
            bestInd = point[1]
    return (bestPoint, bestInd)
'''


def getModel():
    model = nn.Sequential(
        nn.Linear(7, 13),
        nn.ReLU(),
        nn.Linear(13, 2),
    )
    return model

        
def train(model, features, labels, num_epochs):
    # sometimes training is super fast and resulting model sucks, sometimes its super slow and resulting model is good w/ same # of epochs. Not entirely sure why yet.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        for i in range(len(features)):
            running_loss = 0.0
            optimizer.zero_grad()
            outputs = model(torch.FloatTensor([features[i]]))
            loss = criterion(outputs, labels[i])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'current epoch: {epoch}')

def test(model, data):
    assignments = [0] * len(data) #assignments[i] = data[i]'s label
    next_points = []
    next_assignment = 0
    for i in range(len(data) - 1):
        f = data[i][2:]
        #check if this point is close to any point in next_points 
        closePoint,ind  = findClose(f[1:3], next_points)
        print(i, ind)
        if closePoint is not None:
            assignments[i] = assignments[ind]
            pass
        else:
            next_assignment += 1
            assignments[i] = next_assignment

        #if so, assign it the correct id 

        #otherwise, give it new ID
        next_time = data[i + 1][2]
        f = np.append(f, next_time)
        next_point = (model(torch.FloatTensor(f)), i)
        next_points.append((next_point[0].detach().numpy(), i))
    features = data[:,2:]
    labels = data[:,1]
    plotVesselTracks(features[:,[2,1]], np.array(assignments))
    plotVesselTracks(features[:,[2,1]], labels)





def findClose(points, x, y, time, xdir, ydir, speed, timetable):
    #for other_point in points:
    dps = speed  / 36000 #deg per sec

    for t in range(1, 600):
        new_x = x + xdir * dps * t
        new_y = y + ydir * dps * t
        #if no points at this time, give up
        if (float(time + t) not in timetable.keys()):
            continue
        possible_points = timetable[time + t]
        for otherpoint in possible_points:
            dist = distance.euclidean((new_x, new_y), (otherpoint[3], otherpoint[4])) 
            if (dist < .019):
                return otherpoint
                #print(f"match found at t={t}: start: ({x}, {y}), predicted: ({new_x}, {new_y}), acutal: ({otherpoint[3]}, {otherpoint[4]}), dist: {dist}")
    return None




#START IND = 1095
#END IND = 3140
# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    from utils import loadData, plotVesselTracks
    timetable = {}
    points = []
    data = loadData('set2.csv')
    data = preprocess(data)
    for i in range(len(data)):
        if data[i][2] not in timetable.keys():
            timetable[data[i][2]] = []
        timetable[float(data[i][2])].append(data[i])

    start_point = data[1095]
    end_point = data[3140]
    print("Ending time", end_point[2])
    cur_point = start_point
    while (cur_point[2] < end_point[2]): #np.array(cur_point) != np.array(end_point)).any():
        mag = data[i][5]
        xdir = data[i][6]
        ydir = data[i][7]
        next_point = findClose(data, cur_point[3], cur_point[4], cur_point[2], xdir, ydir, mag, timetable)
        if (np.array_equal(next_point, cur_point)):
            print("no next point found")
            break
        elif next_point is not None:
            #print("REASSIGNEMNT OF CURRENT")
            points.append(cur_point)
            cur_point = next_point
            #print(next_point)
            #print(next_point[3], next_point[4], next_point[2])
        else:
            print("no next point found")
            break
    print("terminated")

    #findClose(data, data[i][3], data[i][4], data[i][2], xdir, ydir, mag, timetable)
    #findClose(data)







    

        





# %%
