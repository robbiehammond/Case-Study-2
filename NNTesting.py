# -*- coding: utf-8 -*-
"""
Vessel prediction using k-means clustering on standardized features. If the
number of vessels is not specified, assume 20 vessels.

@author: Kevin S. Xu
"""
import random
from torch import nn
from utils import split_on_VID
import torch

def modelTesting(features, labels, test_features, test_label):
    model = nn.Sequential(
        nn.Linear(1, 10),
        nn.ReLU(),
        nn.Linear(10, 4),
    )
    #trainFeatures = torch.FloatTensor(features)
    #trainFeatures = trainFeatures.to(torch.float32)
    #trainLabels = torch.FloatTensor(labels)
    #trainLabels = trainLabels.to(torch.float32)

    # sometimes training is super fast and resulting model sucks, sometimes its super slow and resulting model is good w/ same # of epochs. Not entirely sure why yet.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()
    for epoch in range(10):
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
    print(f'In format of [latitude, longitude, SOG, COG]\npredicted: {predicted}\nactual: {test_label}')

        



# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    
    from utils import loadData, plotVesselTracks
    data = loadData('set1.csv')
    new_data = split_on_VID(data)
    #features = data[:,2:]
    #labels = data[:,1]
    num_vessels = len(new_data)
    randomVID = random.choice(list(new_data.items()))

    # features = time, labels = location
    features = []
    labels = []
    for i in range(len(randomVID[1]) - 1):
        l = torch.FloatTensor(randomVID[1][i][1:])
        labels.append(l)
        f = randomVID[1][i][0]
        features.append(f)

    # last one for testing: obv not a great form of testing but works as sanity check
    last_sample = randomVID[1][len(randomVID[1]) - 1]
    test_feature = last_sample[0]
    test_label = last_sample[1:]
    modelTesting(features, labels, test_feature, test_label)




