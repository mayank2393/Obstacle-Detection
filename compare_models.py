import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cnn_results = pd.read_csv('Results/cnn_test_result.csv', header=None)
rcnn_results = pd.read_csv('Results/rcnn_test_result.csv', header=None)

ground_truth = pd.read_csv('Dataset/ground-truth-test.csv', sep=',', header=None)

from functions import overlapScore
cnn_score, _ = overlapScore(cnn_results.values, ground_truth.values)
rcnn_score, _ = overlapScore(rcnn_results.values, ground_truth.values)

print(f'RCNN Average Overlap Score: {cnn_score/len(cnn_results)}')
print(f'CNN Average Overlap Score: {rcnn_score/len(rcnn_results)}')

plt.figure(figsize=(10, 5))
plt.plot(cnn_results[0], label="RCNN Predicted")
plt.plot(rcnn_results[0], label="CNN Predicted")
plt.title("Comparison of CNN and RCNN Model Predictions")
plt.legend()
plt.show()
