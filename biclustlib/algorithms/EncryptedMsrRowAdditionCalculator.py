import concrete.numpy as cnp
import numpy as np

class EncryptedMsrRowAdditionCalculator:

    configuration = cnp.Configuration(global_p_error=3 / 100_000, verbose=True)
    @cnp.circuit({"data": "encrypted", "data_cols": "encrypted"}, configuration)
    def squared_residues_rows(data: cnp.tensor[cnp.uint16, 10, 5], data_cols: cnp.tensor[cnp.uint16, 10, 5]):
        data_mean = np.sum(data) // data.size
        row_means = np.sum(data_cols, axis=1, keepdims=True) // data_cols.shape[1]
        col_means = np.sum(data, axis=0) // data.shape[0]

        residues_p1 = data_cols - row_means
        residues_p2 = col_means + data_mean
        row_residues = residues_p1 - residues_p2

        row_squared_residues = row_residues ** 2
        return row_squared_residues

    @cnp.circuit({"row_squared_residues": "encrypted"}, configuration)
    def row_msr_calculator(row_squared_residues: cnp.tensor[cnp.uint16, 10, 5]):
        row_msr = np.sum(row_squared_residues, axis=1) // row_squared_residues.shape[1]
        return row_msr

    @cnp.circuit({"data": "encrypted", "data_cols": "encrypted"}, configuration)
    def squared_residues_inverse_rows(data: cnp.tensor[cnp.uint16, 10, 5], data_cols: cnp.tensor[cnp.uint16, 10, 5]):
        data_mean = np.sum(data) // data.size
        row_means = np.sum(data_cols, axis=1, keepdims=True) // data_cols.shape[1]
        col_means = np.sum(data, axis=0) // data.shape[0]

        residues_p1 = row_means - data_cols
        residues_p2 = col_means + data_mean
        inverse_residues = residues_p1 - residues_p2

        row_inverse_squared_residues = inverse_residues ** 2
        return row_inverse_squared_residues

    @cnp.circuit({"row_inverse_squared_residues": "encrypted"}, configuration)
    def msr_inverse_calculator(row_inverse_squared_residues: cnp.tensor[cnp.uint16, 10, 5]):
        msr_inverse = np.sum(row_inverse_squared_residues, axis=1) // row_inverse_squared_residues.shape[1]
        return msr_inverse


