import concrete.numpy as cnp
import numpy as np

class EncryptedMsrCalculator:
    msr_circuit: cnp.Circuit
    row_msr_circuit: cnp.Circuit
    column_msr_circuit: cnp.Circuit

    def __init__(self, inputset):
        @cnp.compiler({"data": "encrypted"})
        def msr_calculator(data):
            data_mean = np.sum(data) // data.size
            row_means = np.sum(data, axis=1, keepdims=True) // data.shape[1]
            col_means = np.sum(data, axis=0) // data.shape[0]

            residues_p1 = data + (-row_means)
            residues_p2 = col_means + data_mean
            residues = residues_p1 + (-residues_p2)

            squared_residues = residues ** 2
            return np.sum(squared_residues) // squared_residues.size

        self.msr_circuit = msr_calculator.compile(inputset)

        @cnp.compiler({"data": "encrypted"})
        def row_msr_calculator(data):
            data_mean = np.sum(data) // data.size
            row_means = np.sum(data, axis=1, keepdims=True) // data.shape[1]
            col_means = np.sum(data, axis=0) // data.shape[0]

            residues_p1 = data + (-row_means)
            residues_p2 = col_means + data_mean
            residues = residues_p1 + (-residues_p2)

            squared_residues = residues ** 2
            return np.sum(squared_residues, axis=1) // squared_residues.shape[1]

        self.row_msr_circuit = row_msr_calculator.compile(inputset)

        @cnp.compiler({"data": "encrypted"})
        def column_msr_calculator(data):
            data_mean = np.sum(data) // data.size
            row_means = np.sum(data, axis=1, keepdims=True) // data.shape[1]
            col_means = np.sum(data, axis=0) // data.shape[0]

            residues_p1 = data + (-row_means)
            residues_p2 = col_means + data_mean
            residues = residues_p1 + (-residues_p2)

            squared_residues = residues ** 2
            return np.sum(squared_residues, axis=0) // squared_residues.shape[0]

        self.column_msr_circuit = column_msr_calculator.compile(inputset)

    # def evaluate(self, sample):
    #     return (
    #         self.msr_circuit.encrypt_run_decrypt(sample),
    #         self.row_msr_circuit.encrypt_run_decrypt(sample),
    #         self.column_msr_circuit.encrypt_run_decrypt(sample),
    #     )


