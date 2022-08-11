"""
Original Cheng and Church Algorithm
===========================
This example shows resulting biclulsters over synthetic data with constant model
"""

import time
from biclustlib.algorithms import ChengChurchAlgorithm
from biclustlib.datasets import synthetic
import numpy as np

m0 = time.perf_counter()

# load synthetic data
data, predicted = synthetic.make_const_data()

# missing value imputation suggested by Cheng and Church
missing = np.where(data < 0.0)
data[missing] = np.random.randint(low=0, high=800, size=len(missing[0]))

# creating an instance of the ChengChurchAlgorithm class and running with the parameters
cca = ChengChurchAlgorithm(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
biclustering = cca.run(data)
print(biclustering)


m1 = time.perf_counter()
print("Time Performance in Original Algorithm: ", round(m1 - m0, 5), "Seconds")


