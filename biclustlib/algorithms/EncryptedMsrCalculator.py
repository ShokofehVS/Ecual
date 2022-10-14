import math
import concrete.numpy as cnp
import numpy as np

class EncryptedMsrCalculator:
    msr_circuit: cnp.Circuit
    row_msr_circuit: cnp.Circuit
    column_msr_circuit: cnp.Circuit

    def __init__(self, inputset):
        def smallest_prime_divisor(n):
            if n % 2 == 0:
                return 2

            for i in range(3, int(np.sqrt(n)) + 1):
                if n % i == 0:
                    return i
            return n

        def mean_of_vector(x):
            assert x.size != 0
            if x.size == 1:
                return x[0]

            group_size = smallest_prime_divisor(x.size)
            if x.size == group_size:
                return np.round(np.sum(x) / x.size).astype(np.int64)

            groups = []
            for i in range(x.size // group_size):
                start = i * group_size
                end = start + group_size
                groups.append(x[start:end])

            mean_of_groups = []
            for group in groups:
                mean_of_groups.append(np.round(np.sum(group) / group_size).astype(np.int64))

            return mean_of_vector(np.array(mean_of_groups))

        def mean_of_vector_rows(x):
            assert x.size != 0
            if x.size == 1:
                return x[0]

            group_size = smallest_prime_divisor(x.size)
            if x.size == group_size:
                return np.round(np.sum(x) / x.size).astype(np.int64)

            groups = []
            for i in range(x.size // group_size):
                start = i * group_size
                end = start + group_size
                groups.append(x[start:end])

            mean_of_groups = []
            for group in groups:
                mean_of_groups.append(np.round(np.sum(group) / group_size).astype(np.int64))

            return mean_of_vector(np.array(mean_of_groups))

        def mean_of_matrix(x):
            return mean_of_vector(x.flatten())

        def mean_of_rows_of_matrix(x):
            means = []
            for i in range(x.shape[0]):
                means.append(mean_of_vector(x[i]))
            return cnp.array(means)

        def mean_of_columns_of_matrix(x):
            means = []
            for i in range(x.shape[1]):
                means.append(mean_of_vector(x[:, i]))
            return cnp.array(means)

        @cnp.compiler({"data": "encrypted"})
        def msr_calculator(data):
            data_mean = mean_of_matrix(data)
            row_means = mean_of_rows_of_matrix(data)
            row_means = row_means.reshape((-1, 1))
            col_means = mean_of_columns_of_matrix(data)

            residues_p1 = data + (-row_means)
            residues_p2 = col_means + data_mean
            residues = residues_p1 + (-residues_p2)

            squared_residues = residues ** 2
            return mean_of_matrix(squared_residues)
            # return residues

        self.msr_circuit = msr_calculator.compile(inputset)

        @cnp.compiler({"data": "encrypted"})
        def row_msr_calculator(data):
            data_mean = mean_of_matrix(data)
            row_means = mean_of_rows_of_matrix(data)
            row_means = row_means.reshape((-1, 1))
            col_means = mean_of_columns_of_matrix(data)

            residues_p1 = data + (-row_means)
            residues_p2 = col_means + data_mean
            residues = residues_p1 + (-residues_p2)

            squared_residues = residues ** 2
            return mean_of_rows_of_matrix(squared_residues)

        self.row_msr_circuit = row_msr_calculator.compile(
            inputset,
            cnp.Configuration(
                dump_artifacts_on_unexpected_failures=False,
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
            ),
        )

        @cnp.compiler({"data": "encrypted"})
        def column_msr_calculator(data):
            data_mean = mean_of_matrix(data)
            row_means = mean_of_rows_of_matrix(data)
            row_means = row_means.reshape((-1, 1))
            col_means = mean_of_columns_of_matrix(data)

            residues_p1 = data + (-row_means)
            residues_p2 = col_means + data_mean
            residues = residues_p1 + (-residues_p2)

            squared_residues = residues ** 2
            return mean_of_columns_of_matrix(squared_residues)

        self.column_msr_circuit = column_msr_calculator.compile(
            inputset,
            cnp.Configuration(
                dump_artifacts_on_unexpected_failures=False,
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=".keys",
            ),
        )

