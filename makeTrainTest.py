import os
import glob
import shutil
import random
import numpy as np

# set paramters
seed = 4

# set the train, validation, test, everything-else split
split = (0.1, 0.05, 0.05, 0.8)

# clean out the directories:
for dir in ["./train/", "./val/", "./test/"]:
    fullpath = glob.glob(dir + "/*/*")
    files = [f for f in fullpath if os.path.isfile(f)]
    for file in files:
        os.remove(file)


# get list of files
fullpath = glob.glob("./base_data/*")
files = [os.path.split(f)[1] for f in fullpath if os.path.isfile(f)]

# file names
cats = [f for f in files if "cat" in str(f)]
dogs = [f for f in files if "dog" in str(f)]
random.Random(seed).shuffle(cats)
random.Random(seed).shuffle(dogs)

# check - should be 12500 of each
print(len(cats))
print(len(dogs))

# work out splits
cat_split = [int(f * len(cats)) for f in split]
print(f"There will be a cat split of {cat_split}")
cat_split = np.cumsum(cat_split)

dog_split = [int(f * len(dogs)) for f in split]
print(f"There will be a dog split of {dog_split}")
dog_split = np.cumsum(dog_split)


# make the train set cats
for f in cats[0 : cat_split[0]]:
    shutil.copy(os.path.join("./base_data", f), os.path.join("./train/train_cats", f))

# make the validation set cats
for f in cats[cat_split[0] : cat_split[1]]:
    shutil.copy(os.path.join("./base_data", f), os.path.join("./val/val_cats", f))

# #make the test set with remaining cats
for f in cats[cat_split[1] : cat_split[2]]:
    shutil.copy(os.path.join("./base_data", f), os.path.join("./test/test_cats", f))

# #make the train set dogs
for f in dogs[0 : dog_split[0]]:
    shutil.copy(os.path.join("./base_data", f), os.path.join("./train/train_dogs", f))

# #make the validation set dogs
for f in dogs[dog_split[0] : dog_split[1]]:
    shutil.copy(os.path.join("./base_data", f), os.path.join("./val/val_dogs", f))

# #make the test set with remaining dogs
for f in dogs[dog_split[1] : dog_split[2]]:
    shutil.copy(os.path.join("./base_data", f), os.path.join("./test/test_dogs", f))
