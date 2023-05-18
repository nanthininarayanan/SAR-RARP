import os
import random


def classdata(dir_path_all):
  class_0 = []
  class_1 = []
  class_2 = []
  class_3 = []
  class_4 = []
  class_5 = []
  class_6 = []
  class_7 = []

  # List of files 
  for dir_path in dir_path_all:
    for path in os.listdir(dir_path):
      if os.path.isfile(os.path.join(dir_path, path)):
          if dir_path[-1] == str(0):
            class_0.append(os.path.join(dir_path, path))
          elif dir_path[-1] == str(1):
            class_1.append(os.path.join(dir_path, path))
          elif dir_path[-1] == str(2):
            class_2.append(os.path.join(dir_path, path))
          elif dir_path[-1] == str(3):
            class_3.append(os.path.join(dir_path, path))
          elif dir_path[-1] == str(4):
            class_4.append(os.path.join(dir_path, path))
          elif dir_path[-1] == str(5):
            class_5.append(os.path.join(dir_path, path))
          elif dir_path[-1] == str(6):
            class_6.append(os.path.join(dir_path, path))
          elif dir_path[-1] == str(7):
            class_7.append(os.path.join(dir_path, path))
            
  random.seed(42)
  random.shuffle(class_0)
  random.shuffle(class_1)
  random.shuffle(class_2)
  random.shuffle(class_3)
  random.shuffle(class_4)
  random.shuffle(class_5)
  random.shuffle(class_6)
  random.shuffle(class_7)

  class0_test, class0_train = class_0[:int(0.2*len(class_0))], class_0[int(0.2*len(class_0)):]
  class1_test, class1_train = class_1[:int(0.2*len(class_1))], class_1[int(0.2*len(class_1)):]
  class2_test, class2_train = class_2[:int(0.2*len(class_2))], class_2[int(0.2*len(class_2)):]
  class3_test, class3_train = class_3[:int(0.2*len(class_3))], class_3[int(0.2*len(class_3)):]
  class4_test, class4_train = class_4[:int(0.2*len(class_4))], class_4[int(0.2*len(class_4)):]
  class5_test, class5_train = class_5[:int(0.2*len(class_5))], class_5[int(0.2*len(class_5)):]
  class6_test, class6_train = class_6[:int(0.2*len(class_6))], class_6[int(0.2*len(class_6)):]
  class7_test, class7_train = class_7[:int(0.2*len(class_7))], class_7[int(0.2*len(class_7)):]

  test_set = class0_test + class1_test + class2_test + class3_test + class4_test + class5_test + class6_test + class7_test
  
  return test_set, class0_train, class1_train, class2_train, class3_train, class4_train, class5_train, class6_train, class7_train

