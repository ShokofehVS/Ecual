import concrete.numpy as cnp
import numpy as np

class EncryptedMsrColAdditionCalculator:

    configuration = cnp.Configuration(global_p_error=3 / 100_000, verbose=True,
                                      dump_artifacts_on_unexpected_failures=False,
                                      enable_unsafe_features=True,
                                      use_insecure_key_cache=True,
                                      insecure_key_cache_location=".keys", )
    @cnp.circuit({"data": "encrypted", "data_rows": "encrypted"}, configuration)
    def squared_residues(data: cnp.tensor[cnp.uint16, 10, 5], data_rows: cnp.tensor[cnp.uint16, 10, 5]):
        data_mean = np.sum(data) // data.size
        row_means = np.sum(data, axis=1, keepdims=True) // data.shape[1]
        col_means = np.sum(data_rows, axis=0) // data_rows.shape[0]

        residues_p1 = data_rows - row_means
        residues_p2 = col_means + data_mean
        col_residues = residues_p1 - residues_p2

        col_squared_residues = col_residues ** 2
        return col_squared_residues

    @cnp.circuit({"col_squared_residues": "encrypted"}, configuration)
    def msr_column_addition_calculator(col_squared_residues: cnp.tensor[cnp.uint16, 10, 5]):
        col_msr = np.sum(col_squared_residues, axis=0) // col_squared_residues.shape[0]
        return col_msr
