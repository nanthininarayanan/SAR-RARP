from keras.utils import np_utils
from utils.read_videos import Videoto3D
from utils.load_data import loaddata
from utils.dataset import classdata
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np

#add path to folder for all video files
dir_path_all = ["drive/MyDrive/BDD/other_clipped/class0","drive/MyDrive/BDD/other_clipped/class1","drive/MyDrive/BDD/other_clipped/class2","drive/MyDrive/BDD/other_clipped/class3","drive/MyDrive/BDD/other_clipped/class4","drive/MyDrive/BDD/other_clipped/class5","drive/MyDrive/BDD/other_clipped/class6","drive/MyDrive/BDD/other_clipped/class7"]

test_set, _, _, _, _, _, _, _, _ = classdata(dir_path_all)

from tqdm import tqdm

depth = 16
img_rows, img_cols, frames = 112, 112, depth
channel = 3
color = True
skip = False

vid3d = Videoto3D(img_rows, img_cols, frames)
nb_classes = 8

#model to be tested
model_file = '/Users/nanthininarayanan/Desktop/Courses/BDD/Code Database/models/RandomForest_8class_Model_9.h5'
cmodel = tf.keras.models.load_model(model_file)

testX, testY = loaddata(test_set, vid3d, color, skip)
testX = testX.reshape((testX.shape[0], frames, img_cols, img_rows, channel))
testX = testX.astype('float32')
_, test_acc = cmodel.evaluate(testX, np_utils.to_categorical(testY,num_classes=nb_classes), verbose=0)
print("Test accuracy for model 9: {:.2f}%".format(test_acc*100))

rounded_predictions = cmodel.predict(testX)

folder_path = '/Users/nanthininarayanan/Desktop/Courses/BDD/Code Database/testpred'
file_name = "Eval_Tensor_9.npy"
file_path = os.path.join(folder_path, file_name)

np.save(file_path, rounded_predictions)

