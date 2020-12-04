




# https://www.geeksforgeeks.org/how-to-convert-an-image-to-numpy-array-and-saveit-to-csv-file-using-python/

import os, sys

images_path = sys.argv[1]


if not os.path.exists(images_path): print("[?] No Such Path!"); sys.exit(1)
# step-1) resize image to 64X64 pixel
# step-2) turn to grayscale
# step-2) convert each into numpy array
# step-3) save all arrays into csv file
# step-4) save into train_x.csv, train_y.csv, test_x.csv and test_y.csv 