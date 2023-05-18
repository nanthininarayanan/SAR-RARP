import cv2
import pandas as pd
import numpy as np
import os

class VideoProcessing:
    def clip_videos(file,vid_file,clipno,vidno):
        X = pd.read_fwf(file,header=None)
        Y = X.to_numpy()
        parts=[]

        for i in range(Y.shape[0]):
            s = Y[i,0];
            if s == 0:
                s=1;
            e = Y[i,1];
            label = Y[i,2];
            for j in range(e//clipno):
                if (s+clipno-1)>e:
                    break;
                else:
                    parts.append((s,s+clipno-1));
                    s=s+clipno;
                    
            cap = cv2.VideoCapture(vid_file)
            ret, frame = cap.read()
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writers = [cv2.VideoWriter(f"/Users/nanthininarayanan/Desktop/Courses/BDD/Code Database/class{label}/vid{vidno}_part{start}-{end}.avi", fourcc, 20.0, (w, h)) for start, end in parts]
            f = 0
            while ret:
                f += 1
                for i, part in enumerate(parts):
                    start, end = part
                    if start <= f <= end:
                        writers[i].write(frame)
                ret, frame = cap.read()
            for writer in writers:
                writer.release()
            cap.release()
            print(start, end, label)
            parts=[]
    
    
    def count_vids(dir_path):
        # count no of videos in the folder

        count = 0
        # Iterate directory
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1
        print('File count:', count)
