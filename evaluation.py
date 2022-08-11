"""
Evaluation of both Cheng and Church Algorithm versions
===========================
This example shows evaluation of resulting biclulsters over synthetic data with constant model
"""
from biclustlib.algorithms import ChengChurchAlgorithm, ecual
from biclustlib.evaluation import clustering_error, csi
from biclustlib.datasets import synthetic
import matplotlib.pyplot as plt
import numpy as np

# load synthetic data
data, predicted = synthetic.make_const_data()

# missing value imputation suggested by Cheng and Church
missing = np.where(data < 0.0)
data[missing] = np.random.randint(low=0, high=800, size=len(missing[0]))

# shape of data
num_rows, num_cols = data.shape

# creating an instance of the ChengChurchAlgorithm and ecual classes and running with the parameters
cca = ChengChurchAlgorithm(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)
ecu = ecual(num_biclusters=5, msr_threshold=300.0, multiple_node_deletion_threshold=1.2)

bicluster_ref = cca.run(data)
bicluster_pre = ecu.run(data)

# evaluation encrypted and non-encrypted Cheng and Church Algorithm
ce_eval = clustering_error(bicluster_pre, bicluster_ref, num_rows, num_cols)
csi_eval = csi(bicluster_pre, bicluster_ref, num_rows, num_cols)

# visualization with matplotlib
plt.bar(ce_eval, csi_eval, color='blue')
plt.title('Comparison of Ecual with CCA')
plt.xlabel('Clustering Error')
plt.ylabel('Campello Soft Index')
plt.savefig('eval_final.png')
plt.show()

