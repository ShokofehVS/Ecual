import concrete.numpy as cnp
import numpy as np

class EncryptedMsrColAdditionCalculator:
    msr_column_circuit: cnp.Circuit

    def __init__(self, inputset):
        @cnp.compiler({"data": "encrypted", "data_rows": "encrypted"})
        def msr_column_addition_calculator(data, data_rows):
            data_mean = np.sum(data) // data.size
            row_means = np.sum(data, axis=1, keepdims=True) // data.shape[1]
            col_means = np.sum(data_rows, axis=0) // data_rows.shape[0]

            residues_p1 = data_rows + (-row_means)
            residues_p2 = col_means + data_mean
            col_residues = residues_p1 + (-residues_p2)

            col_squared_residues = col_residues ** 2
            return np.sum(col_squared_residues, axis=0) // col_squared_residues.shape[0]

        self.msr_column_circuit = msr_column_addition_calculator.compile(
            inputset,
            cnp.Configuration(
                verbose=True,
                dump_artifacts_on_unexpected_failures=False,
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
            ),
        )
