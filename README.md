# SAR-RARP

Action recognition is one of the most essential and increasingly explored task in computer vision. The development of a precise action recognition method on a surgical dataset is particularly challenging but beneficial since it could contribute to the guidance of a surgical robot and surgical education.

We propose a Time Distributed-Convolutional Neural Network-Long Short-Term Memory (TD-CNN-LSTM) model that improves upon the Endo3D CNN-LSTM baseline model via a time-distributed framework pictured below. Our method is end-to-end and has the potential to outperform state-of-the-art models.

![IMG_0047](https://github.com/nanthininarayanan/SAR-RARP/assets/112656894/e35bd13f-9e29-42d2-986e-cffc2d9d0a87)

## Data

The SAR-RARP50 training sets (1 and 2) were used in this project. The SAR-RARP50 paper has not been published yet. The challenge
description is available at: https://www.synapse.org/Synapse:syn27618412/wiki/

## Code

The train file runs and saves the models on a subset of the data. The test file saves the predictions of the model on test set. The evaluate file calculates the accuracy and other metrics of the group of models and also displays the confusion matrix. Random forest algorithm is implemented with this code. The models used can be found in the models folder along with the test predictions in the testpred folder and the true labels in the testlabels folder.

## Software Requirements

The packages that need to be installed to run this code are specified in the packages.txt file in the env folder. 

## Hardware Requirements

GPU computing units are required to run the models.

