from sklearn.utils import resample
import numpy as np

# Random seed fixed so result could be replicated by Reader
np.random.seed(123)
# data to be sampled
data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# Number of divisions needed
num_divisions = 3
list_of_data_divisions = []
for x in range(0, num_divisions):
    sample = resample(data, replace=True, n_samples=4)
    list_of_data_divisions.append(sample)

print("Samples", list_of_data_divisions)
# Output: Samples [[3, 3, 7, 2], [4, 7, 2, 1], [2, 1, 1, 4]]
