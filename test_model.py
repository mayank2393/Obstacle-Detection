import numpy as np
import pandas as pd
import torch
from functions import overlapScore
from cnn_model import cnn_model
from rcnn_model import rcnn_model

def test_model(model, model_name, testX, groundTruth):
    model.eval()
    model.load_state_dict(torch.load(f'Model/{model_name}_model.pth'))
    output = model(torch.Tensor(np.reshape(testX, (len(testX), 1, 100, 100))))
    
    output = output.detach().numpy()
    output = output.astype(int)

    score, _ = overlapScore(output, groundTruth)
    score /= len(testX)
    print(f'{model_name} Test Average overlap score : {score:.6f}')

    np.savetxt(f'Results/{model_name}_test_result.csv', output, delimiter=',')

if __name__ == '__main__':
    testX = pd.read_csv('Dataset/testData.csv', sep=',', header=None)
    groundTruth = pd.read_csv('Dataset/ground-truth-test.csv', sep=',', header=None)
    testX = np.asanyarray(testX)
    groundTruth = np.asarray(groundTruth)

    # Test CNN model
    cnn_model_instance = cnn_model()
    test_model(cnn_model_instance, "cnn", testX, groundTruth)

    # Test RCNN model
    rcnn_model_instance = rcnn_model()
    test_model(rcnn_model_instance, "rcnn", testX, groundTruth)
