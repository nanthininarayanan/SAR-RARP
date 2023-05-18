import numpy as np
import os
import matplotlib
matplotlib.use('AGG')
import numpy as np
from tqdm import tqdm

#function to load videos and labels from the given path to directory
def loaddata(training_set, vid3d, color=False, skip=True):

    X = []
    labels = []
    pbar = tqdm(total=len(training_set))

        
    for v_files in training_set:

        pbar.update(1)
        
        X.append(vid3d.video3d(v_files, color=color, skip=skip))

        labels.append(int(os.path.basename(os.path.dirname(v_files))[-1]))

    pbar.close()

    if color:
        return np.array(X), labels
    else:
        return np.array(X), labels