from utils import loadData, preprocess, findClose
timetable = {}
points = []
data = loadData('set2.csv')
data = preprocess(data)
# Create dictionary for faster data transfer
times = [58540]
for i in range(len(data)):
        if data[i][2] in times:
            print("i", i, 'lat:', data[i][3], 'time: ', data[i][2])