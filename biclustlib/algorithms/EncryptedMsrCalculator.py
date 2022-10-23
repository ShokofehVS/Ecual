import math
import concrete.numpy as cnp
import numpy as np

class EncryptedMsrCalculator():

    @cnp.circuit({"data": "encrypted"})
    def squared_residues(data: cnp.tensor[cnp.uint16, 10, 5]):
        data_mean = np.sum(data) // data.size
        row_means = np.sum(data, axis=1, keepdims=True) // data.shape[1]
        col_means = np.sum(data, axis=0) // data.shape[0]

        residues_p1 = data - row_means
        residues_p2 = col_means + data_mean
        residues = residues_p1 - residues_p2

        squared_residues = residues ** 2
        return squared_residues

    @cnp.circuit({"squared_residues": "encrypted"})
    def msr_calculator(squared_residues: cnp.tensor[cnp.uint16, 10, 5]):
        msr = np.sum(squared_residues) // squared_residues.size
        return msr

    @cnp.circuit({"squared_residues": "encrypted"})
    def row_msr_calculator(squared_residues: cnp.tensor[cnp.uint16, 10, 5]):
        row_msr = np.sum(squared_residues, axis=1) // squared_residues.shape[1]
        return row_msr

    @cnp.circuit({"squared_residues": "encrypted"})
    def column_msr_calculator(squared_residues: cnp.tensor[cnp.uint16, 10, 5]):
        col_msr = np.sum(squared_residues, axis=0) // squared_residues.shape[0]
        return col_msr
