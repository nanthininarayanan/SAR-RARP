import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report,confusion_matrix

folder_path = "/Users/nanthininarayanan/Desktop/Courses/BDD/Code Database/testpred" 

# Get a list of all file paths in the folder
file_paths = glob.glob(os.path.join(folder_path, "*"))

test_tensors = []

# Iterate over the file paths and load the test tensors
for file_path in file_paths:
    tensor = np.load(file_path)
    test_tensors.append(tensor)

# Stack the arrays along the third axis (i.e., axis=2)
stacked_array = np.stack((test_tensors[0], test_tensors[1], test_tensors[2], test_tensors[3], test_tensors[4], test_tensors[5], test_tensors[6], test_tensors[7], test_tensors[8], test_tensors[9]), axis=2)

# Calculate the average along the third axis (i.e., axis=2)
average_array = np.mean(stacked_array, axis=2)

# Print the shape of the resulting array to verify it is of size (1, 8)
print(average_array.shape)

predicted_labels = np.argmax(average_array, axis=1)

#load labels
labels_file = '/Users/nanthininarayanan/Desktop/Courses/BDD/Code Database/testlabels/Test_labels.npy'
test_labels = np.load(labels_file)
# Count the number of correct predictions
num_correct = np.sum(predicted_labels == test_labels)

# Calculate the accuracy
accuracy = num_correct / len(test_labels)

# Print the accuracy
print(f'Accuracy: {accuracy:.2%}')

#Evaluate performance 
print(classification_report(test_labels, predicted_labels))
Acc= (np.count_nonzero(test_labels==predicted_labels))/len(predicted_labels)*100
print('Acc:',Acc)
#Display confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)
print(cm)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
