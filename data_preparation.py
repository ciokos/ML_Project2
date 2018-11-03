import csv
import numpy as np

datasetpath = "Bike-Sharing-Dataset"

file = open(datasetpath + "/day.csv")
daydata = csv.reader(file, delimiter=',')

row1 = next(daydata)

attribute_names = row1[2:]
X = []
for row in daydata:
    X.append(row[2:])

X = np.array(X).astype(float)

print(attribute_names)









