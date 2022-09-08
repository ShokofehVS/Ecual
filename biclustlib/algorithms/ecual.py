from ._base import BaseBiclusteringAlgorithm
from ..models import Bicluster, Biclustering
from sklearn.utils.validation import check_array
from biclustlib.algorithms.EncryptedMsrCalculator import EncryptedMsrCalculator
import numpy as np
import concrete.numpy as cnp
import time



class ecual(BaseBiclusteringAlgorithm):
    """Secured Cheng and Church's Algorithm (CCA)

    ecual searches for maximal submatrices with a Mean Squared Residue value below a pre-defined threshold
        by Homomorphic Encryption operations


    Parameters
    ----------
    num_biclusters : int, default: 5
        Number of biclusters to be found.

    msr_threshold : float, default: 300
        Maximum mean squared residue accepted (delta parameter in the original paper).

    multiple_node_deletion_threshold : float, default: 1.2
        Scaling factor to remove multiple rows or columns (alpha parameter in the original paper).

    data_min_cols : int, default: 100
        Minimum number of dataset columns required to perform multiple column deletion.
    """

    def __init__(self, num_biclusters=5, msr_threshold=300, multiple_node_deletion_threshold=1.2, data_min_cols=100):
        self.num_biclusters = num_biclusters
        self.msr_threshold = msr_threshold
        self.multiple_node_deletion_threshold = multiple_node_deletion_threshold
        self.data_min_cols = data_min_cols

    def run(self, data):
        """Compute biclustering.

        Parameters
        ----------
        data : numpy.ndarray
        """

        data = check_array(data, dtype=np.double, copy=True)
        self._validate_parameters()

        num_rows, num_cols = data.shape
        min_value = np.min(data)
        max_value = np.max(data)
        msr_thr = self.msr_threshold

        biclusters = []

        for i in range(self.num_biclusters):
            rows = np.ones(num_rows, dtype=np.bool)
            cols = np.ones(num_cols, dtype=np.bool)

            self._multiple_node_deletion(data, rows, cols, msr_thr)
            self._single_node_deletion(data, rows, cols, msr_thr)
            self._node_addition(data, rows, cols)

            row_indices = np.nonzero(rows)[0]
            col_indices = np.nonzero(cols)[0]

            if len(row_indices) == 0 or len(col_indices) == 0:
                break

            # masking matrix values
            if i < self.num_biclusters - 1:
                bicluster_shape = (len(row_indices), len(col_indices))
                data[row_indices[:, np.newaxis], col_indices] = np.random.uniform(low=min_value, high=max_value,
                                                                                  size=bicluster_shape)

            biclusters.append(Bicluster(row_indices, col_indices))

        return Biclustering(biclusters)

    def _single_node_deletion(self, data, rows, cols, msr_thr):
        """Performs the single row/column deletion step (this is a direct implementation of the Algorithm 1 described in
        the original paper)"""
        msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)
        while msr > msr_thr:
            self._single_deletion(data, rows, cols, row_msr, col_msr)
            msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)

    def _single_deletion(self, data, rows, cols, row_msr, col_msr):
        """Deletes a row or column from the bicluster being computed."""
        row_indices = np.nonzero(rows)[0]
        col_indices = np.nonzero(cols)[0]

        row_max_msr = np.argmax(row_msr)
        col_max_msr = np.argmax(col_msr)

        if row_msr[row_max_msr] >= col_msr[col_max_msr]:
            row2remove = row_indices[row_max_msr]
            rows[row2remove] = False
        else:
            col2remove = col_indices[col_max_msr]
            cols[col2remove] = False

    def _multiple_node_deletion(self, data, rows, cols, msr_thr):
        """Performs the multiple row/column deletion step (this is a direct implementation of the Algorithm 2 described in
        the original paper)"""
        msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)

        stop = True if msr <= msr_thr else False

        while not stop:
            cols_old = np.copy(cols)
            rows_old = np.copy(rows)

            row_indices = np.nonzero(rows)[0]
            rows2remove = row_indices[np.where(row_msr > self.multiple_node_deletion_threshold * msr)]
            rows[rows2remove] = False

            if len(cols) >= self.data_min_cols:
                msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)
                col_indices = np.nonzero(cols)[0]
                cols2remove = col_indices[np.where(col_msr > self.multiple_node_deletion_threshold * msr)]
                cols[cols2remove] = False

            msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)

            # Tests if the new MSR value is smaller than the acceptable MSR threshold.
            # Tests if no rows and no columns were removed during this iteration.
            # If one of the conditions is true the loop must stop, otherwise it will become an infinite loop.
            if msr <= msr_thr or (np.all(rows == rows_old) and np.all(cols == cols_old)):
                stop = True

    def _node_addition(self, data, rows, cols):
        """Performs the row/column addition step (this is a direct implementation of the Algorithm 3 described in
        the original paper)"""
        stop = False
        while not stop:
            cols_old = np.copy(cols)
            rows_old = np.copy(rows)

            msr, _, _ = self._calculate_msr(data, rows, cols)
            col_msr = self._calculate_msr_col_addition(data, rows, cols)
            cols2add = np.where(col_msr <= msr)[0]
            cols[cols2add] = True

            msr, _, _ = self._calculate_msr(data, rows, cols)
            row_msr, row_inverse_msr = self._calculate_msr_row_addition(data, rows, cols)
            rows2add = np.where(np.logical_or(row_msr <= msr, row_inverse_msr <= msr))[0]
            rows[rows2add] = True

            if np.all(rows == rows_old) and np.all(cols == cols_old):
                stop = True
    def cnp_datamean(self, data):
        return np.sum(data) // data.size

    def cnp_rowmean(self, data):
        row_means = np.sum(data, axis=1) // data.shape[1]
        return row_means.reshape((data.shape[0], 1))

    def cnp_colmean(self, data):
        return np.sum(data, axis=0) // data.shape[0]

    def cnp_add(self, value1, value2):
        return value1 + value2

    def cnp_square(self, value1):
        return value1 ** 2

    def cnp_sub(self, value1, value2):
        return np.subtract(value1, value2)

    def getEnc_mean(self, inputset, data):
        compiler = cnp.Compiler(self.cnp_datamean, {"data": "encrypted"})
        circuit = compiler.compile(inputset)
        data = data.astype('uint8')
        circuit.keygen()
        public_args = circuit.encrypt(data)
        encrypted_datamean = circuit.run(public_args)
        decrypted_result = circuit.decrypt(encrypted_datamean)
        return decrypted_result

    def getEnc_rowmean(self, data):
        compiler = cnp.Compiler(self.cnp_rowmean, {"data": "encrypted"})
        inputset = [np.random.randint(0, 30, size=(2, 2), dtype=np.uint8) for _ in range(10)]
        circuit = compiler.compile(inputset)
        data = data.astype('uint8')
        circuit.keygen()
        public_args = circuit.encrypt(data)
        encrypted_rowmean = circuit.run(public_args)
        decrypted_result = circuit.decrypt(encrypted_rowmean)
        return decrypted_result

    def getEnc_colmean(self, inputset, data):
        compiler = cnp.Compiler(self.cnp_colmean, {"data": "encrypted"})
        circuit = compiler.compile(inputset)
        data = data.astype('uint8')
        circuit.keygen()
        public_args = circuit.encrypt(data)
        encrypted_colmean = circuit.run(public_args)
        decrypted_result = circuit.decrypt(encrypted_colmean)
        return decrypted_result

    # def getEnc_reshape(self, data, sub_data):
    #     compiler = cnp.Compiler(self.cnp_reshape, {"data": "encrypted", "sub_data": "clear"})
    #     inputset = [
    #         ([np.random.randint(0, 30, size=(2, ), dtype=np.uint8) for _ in range(10)],
    #          1)
    #     ]
    #     circuit = compiler.compile(inputset)
    #     data = data.astype('uint8')
    #     sub_data = sub_data.astype('uint8')
    #     circuit.keygen()
    #     public_args = circuit.encrypt(data, sub_data)
    #     encrypted_reshape = circuit.run(public_args)
    #     return encrypted_reshape
    def getEnc_colmean_addition(self, data, rows):
        compiler = cnp.Compiler(self.cnp_colmean, {"data": "encrypted"})
        inputset = [np.random.randint(0, 30, size=(2, 2)) for _ in range(10)]
        circuit = compiler.compile(inputset)
        sample = data[rows]
        # sample = sample.astype(int)
        sample = sample.astype('uint8')
        circuit.keygen()
        public_args = circuit.encrypt(sample)
        encrypted_colmean = circuit.run(public_args)
        return encrypted_colmean

    def getEnc_rowmean_addition(self, data, cols):
        compiler = cnp.Compiler(self.cnp_colmean, {"data": "encrypted"})
        inputset = [np.random.randint(0, 30, size=(2, 2)) for _ in range(10)]
        circuit = compiler.compile(inputset)
        sample = data[:, cols]
        sample = sample.astype(int)
        circuit.keygen()
        public_args = circuit.encrypt(sample)
        encrypted_colmean = circuit.run(public_args)
        return encrypted_colmean

    def getEnc_addition(self, vec1, vec2):
        compiler = cnp.Compiler(self.cnp_add, {"value1": "encrypted", "value2": "encrypted"})
        inputset = [
            ((np.random.randint(0, 30, size=(2, )) for _ in range(10)), (range(100)))
        ]
        circuit = compiler.compile(inputset)
        sample = [
            (vec1, vec2),
        ]
        # sample = sample.astype(int)
        circuit.keygen()
        public_args = circuit.encrypt(sample)
        encrypted_addition = circuit.run(public_args)
        return encrypted_addition

    def getEnc_subtraction_prt1(self, vec1, vec2):
        compiler = cnp.Compiler(self.cnp_sub, {"value1": "encrypted", "value2": "encrypted"})
        inputset = [
            ([np.random.randint(0, 30, size=(2, 2), dtype=np.uint8) for _ in range(10)],
             [np.random.randint(0, 30, size=(2, 1), dtype=np.uint8) for _ in range(10)])
        ]
        circuit = compiler.compile(inputset)
        sample = [
            (vec1, vec2)
        ]
        # sample = sample.astype(int)
        circuit.keygen()
        public_args = circuit.encrypt(sample)
        encrypted_addition = circuit.run(public_args)
        return encrypted_addition

    def getEnc_square(self, vec1):
        compiler = cnp.Compiler(self.cnp_square(), {"value1": "encrypted"})
        inputset = [np.random.randint(0, 30, size=(2, 2)) for _ in range(10)]
        circuit = compiler.compile(inputset)
        sample = [vec1]
        # sample = sample.astype(int)
        circuit.keygen()
        public_args = circuit.encrypt(sample)
        encrypted_square = circuit.run(public_args)
        return encrypted_square

    def getEnc_msr(self, squared_residues):
        compiler = cnp.Compiler(self.cnp_datamean, {"data": "encrypted"})
        inputset = [np.random.randint(0, 30, size=(2, 2)) for _ in range(10)]
        circuit = compiler.compile(inputset)
        sample = squared_residues
        sample = sample.astype(int)
        circuit.keygen()
        public_args = circuit.encrypt(sample)
        encrypted_datamean = circuit.run(public_args)
        return encrypted_datamean

    def getEnc_rowmsr(self, squared_residues):
        compiler = cnp.Compiler(self.cnp_rowmean, {"data": "encrypted"})
        inputset = [np.random.randint(0, 30, size=(2, 2)) for _ in range(10)]
        circuit = compiler.compile(inputset)
        sample = squared_residues
        sample= sample.astype(int)
        circuit.keygen()
        public_args = circuit.encrypt(sample)
        encrypted_rowmean = circuit.run(public_args)
        return encrypted_rowmean

    def getEnc_colmsr(self, squared_residues):
        compiler = cnp.Compiler(self.cnp_colmean, {"data": "encrypted"})
        inputset = [np.random.randint(0, 30, size=(2, 2)) for _ in range(10)]
        circuit = compiler.compile(inputset)
        sample = squared_residues
        sample = sample.astype(int)
        circuit.keygen()
        public_args = circuit.encrypt(sample)
        encrypted_colmean = circuit.run(public_args)
        return encrypted_colmean

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

    # @cnp.compiler({"data": "encrypted"})
    def _calculate_msr(self, data, rows, cols):
        """Calculate the mean squared residues of the rows, of the columns and of the full data matrix."""
        #
        # inputset = [np.random.randint(0, 5, size=(2, 2)) for _ in range(10)]
        sample = data[rows][:, cols]
        sample = sample.astype('uint8')
        #
        # msr_circuit: cnp.Circuit
        # row_msr_circuit: cnp.Circuit
        # column_msr_circuit: cnp.Circuit
        #
        # msr_circuit = self.msr_calculator.compile(inputset)
        # row_msr_circuit = self.row_msr_calculator.compile(inputset)
        # column_msr_circuit = self.column_msr_calculator.compile(inputset)
        #
        # return(
        #     msr_circuit.encrypt_run_decrypt(input),
        #     row_msr_circuit.encrypt_run_decrypt(input),
        #     column_msr_circuit.encrypt_run_decrypt(input),
        # )
        inputset = [np.random.randint(0, 5, size=(2, 2)) for _ in range(10)]
        msrcalculator = EncryptedMsrCalculator(inputset)
        msr = msrcalculator.msr_circuit.encrypt_run_decrypt(sample)
        row_msr = msrcalculator.row_msr_circuit.encrypt_run_decrypt(sample)
        col_msr = msrcalculator.column_msr_circuit.encrypt_run_decrypt(sample)

        return msr, row_msr, col_msr


        # msrcalculator.evaluate(sample)
        # return msr, row_msr, col_msr

        # inputset = [np.random.randint(0, 30, size=(2, 2)) for _ in range(10)]
        # data_mean = np.mean(sub_data)
        # row_means = np.mean(sub_data, axis=1)
        # col_means = np.mean(sub_data, axis=0)
        # row_means = row_means[:, np.newaxis]
        # data_mean = self.getEnc_mean(inputset, sub_data)
        #
        # row_means = self.getEnc_rowmean(sub_data)
        #
        # col_means = self.getEnc_colmean(inputset, sub_data)

        # residue_prt1 = self.getEnc_subtraction_prt1(sub_data, row_means)
        # residue_prt2 = self.getEnc_addition(col_means, data_mean)
        # residues = self.getEnc_subtraction(residue_prt1, residue_prt2)

        # squared_residues = residues * residues
        # squared_residues = residues ** 2
        # squared_residues = residues ** 2
        #
        # # msr = np.mean(squared_residues)
        # msr = self.getEnc_msr(squared_residues)
        # # row_msr = np.mean(squared_residues, axis=1)
        # row_msr = self.getEnc_rowmsr(squared_residues)
        # # col_msr = np.mean(squared_residues, axis=0)
        # col_msr = self.getEnc_colmsr(squared_residues)


    def _calculate_msr_col_addition(self, data, rows, cols):
        """Calculate the mean squared residues of the columns for the node addition step."""

        # sub_data = data[rows][:, cols]

        # cf = concretefun()
        # compiler = hnp.NPFHECompiler(cf.submatrix, {"x": "encrypted"})
        # inputset = [(col_means.astype(dtype=np.uint16), data_mean.astype(dtype=np.uint16))]
        # circuit = compiler.compile_on_inputset(inputset)
        # print(compiler)
        # print(circuit.encrypt_run_decrypt([(col_means.astype(dtype=np.uint16), data_mean.astype(dtype=np.uint16))]))

        sub_data_rows = data[rows]

        # data_mean = np.mean(sub_data)
        data_mean = self.getEnc_mean(data,rows,cols)
        # row_means = np.mean(sub_data, axis=1)
        row_means = self.getEnc_rowmean(data,rows,cols)
        # col_means = np.mean(sub_data_rows, axis=0)
        col_means = self.getEnc_colmean_addition(data,rows)


        # col_residues = sub_data_rows - row_means[:, np.newaxis] - col_means + data_mean
        col_residues = sub_data_rows - row_means[:, np.newaxis] - col_means + data_mean
        # col_squared_residues = col_residues * col_residues
        col_squared_residues = col_residues ** 2
        # col_msr = np.mean(col_squared_residues, axis=0)
        col_msr = self.getEnc_colmsr(col_squared_residues)

        return col_msr

    def _calculate_msr_row_addition(self, data, rows, cols):
        """Calculate the mean squared residues of the rows and of the inverse of the rows for
        the node addition step."""

        sub_data = data[rows][:, cols]
        sub_data_cols = data[:, cols]

        # data_mean = np.mean(sub_data)
        data_mean = self.getEnc_mean(data,rows,cols)
        # row_means = np.mean(sub_data_cols, axis=1)
        row_means = self.getEnc_rowmean_addition(data,cols)
        # col_means = np.mean(sub_data, axis=0)
        col_means = self.getEnc_colmean(data,rows,cols)

        row_residues = sub_data_cols - row_means[:, np.newaxis] - col_means + data_mean
        # row_squared_residues = row_residues * row_residues
        row_squared_residues = row_residues ** 2
        row_msr = self.getEnc_rowmsr(row_squared_residues)

        inverse_residues = -sub_data_cols + row_means[:, np.newaxis] - col_means + data_mean
        row_inverse_squared_residues = inverse_residues ** 2
        # row_inverse_msr = np.mean(row_inverse_squared_residues, axis=1)
        row_inverse_msr = self.getEnc_rowmsr(row_inverse_squared_residues)

        return row_msr, row_inverse_msr

    def _validate_parameters(self):
        if self.num_biclusters <= 0:
            raise ValueError("num_biclusters must be > 0, got {}".format(self.num_biclusters))

        if self.msr_threshold != 'estimate' and self.msr_threshold < 0.0:
            raise ValueError("msr_threshold must be equal to 'estimate' or a numeric value >= 0.0, got {}".format(
                self.msr_threshold))

        if self.multiple_node_deletion_threshold < 1.0:
            raise ValueError(
                "multiple_node_deletion_threshold must be >= 1.0, got {}".format(self.multiple_node_deletion_threshold))

        if self.data_min_cols < 100:
            raise ValueError("data_min_cols must be >= 100, got {}".format(self.data_min_cols))

