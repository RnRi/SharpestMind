import numpy as np
import pandas as pd

# Read CSV
task_file = 'task_data.csv'
with open(task_file, mode = 'rb') as f:
    data = pd.read_csv(f)
    
# Extract names of each column (using pandas)
headers = np.array(list(data.columns.values))
names = headers[2:]
print ("Feature names shape is {}".format(names.shape))

# Extract features (using pandas and numpy)
np_array = data.as_matrix()
X = np_array[:,2:]
print ("Features shape is {}".format(X.shape))

# Extract labels (using pandas)
Y = data['class_label'].as_matrix()
print ("Labels shape is {}".format(Y.shape))

from sklearn.ensemble import RandomForestRegressor


rf = RandomForestRegressor()
# Fit the model
rf.fit(X, Y)

# Sort in-place from highest to lowest
rank = sorted(zip(map(lambda x: round(x, 2),
                      rf.feature_importances_), names), reverse=True)
print("Features and Predictive Scores:")
for el in rank: print(el)


import matplotlib.pyplot as plt


# Save the labels and their importance scores separately
score = list(zip(*rank))[0]
sensors = list(zip(*rank))[1]

x_pos = np.arange(len(sensors))
fig = plt.figure(figsize=(14, 8))
plt.bar(x_pos, score, align='center')
plt.xticks(x_pos, sensors)
plt.ylabel('Importance Scores')
plt.show()
