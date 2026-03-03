import numpy as np
# y = np.load("../data/processed/y.npy")
# print(np.unique(y))
# print("Number of classes:", len(np.unique(y)))


y = np.load("../data/processed/y.npy")
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))