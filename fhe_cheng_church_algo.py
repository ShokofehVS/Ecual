"""
Cheng and Church Algorithm Homomorphically
===========================
This example shows resulting biclulsters over synthetic data with constant model
"""
import time
from biclustlib.algorithms import ecual
from biclustlib.datasets import synthetic
import numpy as np

m0 = time.perf_counter()

# load synthetic data
data, predicted = synthetic.make_const_data()

# missing value imputation suggested by Cheng and Church
missing = np.where(data < 0.0)
data[missing] = np.random.randint(low=0, high=800, size=len(missing[0]))

# creating an instance of the ecual class and running with the parameters
ecual_ins = ecual(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
biclustering = ecual_ins.run(data)
print(biclustering)

m1 = time.perf_counter()
print("Time Performance in Calculating Homomorphically: ", m1 - m0, "Seconds")


