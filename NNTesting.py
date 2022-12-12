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
        if (d < .08 and d < minDist):
            minDist = d 
            bestPoint = otherPos
            bestInd = point[1]
    return (bestPoint, bestInd)



def modelTesting(features, labels, test_features, test_label):
    model = nn.Sequential(
        nn.Linear(7, 12),
        nn.ReLU(),
        nn.Linear(12, 2),
    )
    #trainFeatures = torch.FloatTensor(features)
    #trainFeatures = trainFeatures.to(torch.float32)
    #trainLabels = torch.FloatTensor(labels)
    #trainLabels = trainLabels.to(torch.float32)

    # sometimes training is super fast and resulting model sucks, sometimes its super slow and resulting model is good w/ same # of epochs. Not entirely sure why yet.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()
    for epoch in range(15):
        for i in range(len(features)):
            running_loss = 0.0
            optimizer.zero_grad()
            outputs = model(torch.FloatTensor([features[i]]))
            loss = criterion(outputs, labels[i])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'current epoch: {epoch}')
    predicted = model(torch.FloatTensor([test_features])).tolist()
    print(f'In format of [latitude, longitude]\npredicted: {predicted}\nactual: {test_label}')
    return model

        



# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    from utils import loadData, plotVesselTracks
    data = loadData('set1.csv')
    data = preprocess(data)
    new_data = split_on_VID(data)
    #features = data[:,2:]
    #labels = data[:,1]
    num_vessels = len(new_data)
    randomVID = random.choice(list(new_data.items()))
    closests, inds = closest_point_map(data) # map the closest point Y to point X for all X



    # features = (cur time, lat, long, SOG, cos(COG), sin(COG), time to pred)
    features = []
    labels = []
    for i in range(len(randomVID[1]) - 2):
        l = torch.FloatTensor(randomVID[1][i + 1][1:3])
        labels.append(l)
        f = randomVID[1][i][0:]
        f.append(randomVID[1][i + 1][0])
        features.append(f)

    # last two for testing (take second to last point of a ship, predict final, compare with actual final): obv not a great form of testing but works as sanity check
    last_sample = randomVID[1][len(randomVID[1]) - 1]
    second_to_last_sample = randomVID[1][len(randomVID[1]) - 2]
    test_feature = second_to_last_sample[0:]
    test_feature.append(last_sample[0])
    test_label = last_sample[1:3]
    model = modelTesting(features, labels, test_feature, test_label)

    data2 = loadData('set2.csv')
    data2 = preprocess(data2)
    sorted(data2, key=lambda e : e[1])

    assignments = [0] * len(data2) #assignments[i] = data[i]'s label
    next_points = []
    next_assignment = 0
    for i in range(len(data2) - 1):
        f = data2[i][2:]
        #check if this point is close to any point in next_points 
        closePoint,ind  = findClose(f[1:3], next_points)
        print(i, ind)
        if closePoint is not None:
            prev_assignment = assignments[ind]
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
    features = data2[:,2:]
    labels = data2[:,1]
    plotVesselTracks(features[:,[2,1]], np.array(assignments))




    

        




