import os
from params import *

folders = (feats_train_folder, labels_train_folder, feats_test_folder,
labels_test_folder, perturb_folder, scalers_folder)
for folder in folders:
    if not os.path.exists(folder):
	os.makedirs(folder)
