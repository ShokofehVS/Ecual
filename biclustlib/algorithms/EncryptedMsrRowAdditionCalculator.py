import concrete.numpy as cnp
import numpy as np

class EncryptedMsrRowAdditionCalculator:
    msr_row_circuit: cnp.Circuit
    msr_inverse_circuit: cnp.Circuit

    def __init__(self, inputset):
        @cnp.compiler({"data": "encrypted", "data_cols": "encrypted"})
        def msr_row_addition_calculator(data, data_cols):
            data_mean = np.sum(data) // data.size
            row_means = np.sum(data_cols, axis=1, keepdims=True) // data_cols.shape[1]
            col_means = np.sum(data, axis=0) // data.shape[0]

            residues_p1 = data_cols + (-row_means)
            residues_p2 = col_means + data_mean
            row_residues = residues_p1 + (-residues_p2)

            row_squared_residues = row_residues ** 2
            return np.sum(row_squared_residues, axis=1) // row_squared_residues.shape[1]

        self.msr_row_circuit = msr_row_addition_calculator.compile(
            inputset,
            cnp.Configuration(
                verbose=True,
                dump_artifacts_on_unexpected_failures=False,
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
            ),
        )


        @cnp.compiler({"data": "encrypted", "data_cols": "encrypted"})
        def msr_inverse_calculator(data, data_cols):
            data_mean = np.sum(data) // data.size
            row_means = np.sum(data_cols, axis=1, keepdims=True) // data_cols.shape[1]
            col_means = np.sum(data, axis=0) // data.shape[0]

            residues_p1 = (-data_cols) + row_means
            residues_p2 = col_means + data_mean
            inverse_residues = residues_p1 + (-residues_p2)

            row_inverse_squared_residues = inverse_residues ** 2
            return np.sum(row_inverse_squared_residues, axis=1) // row_inverse_squared_residues.shape[1]

        self.msr_inverse_circuit = msr_inverse_calculator.compile(
            inputset,
            cnp.Configuration(
                verbose=True,
                dump_artifacts_on_unexpected_failures=False,
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
            ),
        )


