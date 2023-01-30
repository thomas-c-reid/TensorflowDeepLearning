import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

# Using pickle to open dataset we created in this file
# pickle_in = open("X.pickle", "rb")
# X = pickle.load(pickle_in)
# print(X[5])

# Reference to directory images are saved in 
DATADIR = "kagglecatsanddogs_5340/PetImages"
CATEGORIES = ["Dog", "Cat"]

# Reshape the images to smaller img size
IMG_SIZE = 50


training_data = []

# Function to turn all images into an array with greyscale vector and classification before storing them in "training_data"
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            # test_Image = cv2.imread(os.path.join(path,img))
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

# Runs function we just created
create_training_data()

# Shuffles images
random.shuffle(training_data)



# Adds Features and labels to new arrays called X and Y
X = []
Y = []
for features, label in training_data:
    X.append(features)
    Y.append(label)

# Convert X into numpy array
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Using pickle to save X so dataset can be reused
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close

# Using pickle to save Y so dataset can be reused
pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close