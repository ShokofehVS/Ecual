from ._base import BaseBiclusteringAlgorithm
from ..models import Bicluster, Biclustering
from sklearn.utils.validation import check_array
from biclustlib.algorithms.EncryptedMsrCalculator import EncryptedMsrCalculator
from biclustlib.algorithms.EncryptedMsrColAdditionCalculator import EncryptedMsrColAdditionCalculator
from biclustlib.algorithms.EncryptedMsrRowAdditionCalculator import EncryptedMsrRowAdditionCalculator
from biclustlib.algorithms.rows2remove import rows2remove
from biclustlib.algorithms.cols2remove import cols2remove
from biclustlib.algorithms.cols2add import cols2add
from biclustlib.algorithms.rowsaddition import rowsaddition
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

        # Without FHE
        while msr > msr_thr:
            self._single_deletion(data, rows, cols, row_msr, col_msr)
            msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)

        # With FHE
        """  inputset = [np.random.randint(0, 5, size=(2, 2)) for _ in range(10)]
        sample = msr
        singlenodedeletion = SingleNodeDeletion(inputset)
        comparison = singlenodedeletion.comparison.encrypt_run_decrypt(sample)
        while comparison:
            self._single_deletion(data, rows, cols, row_msr, col_msr)
            msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)"""

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

            # without FHE
            row_indices = np.nonzero(rows)[0]
            rows2remove = row_indices[np.where(row_msr > self.multiple_node_deletion_threshold * msr)]
            rows[rows2remove] = False

            # with FHE
            """multiple_row_deletion = rows2remove()
            rows = multiple_row_deletion.multiple_rows_deletion.encrypt_run_decrypt(data.astype('uint16'),
                                                                                     row_msr.astype('uint16'),
                                                                                     msr.astype('uint16'))
"""
            if len(cols) >= self.data_min_cols:
                msr, row_msr, col_msr = self._calculate_msr(data, rows, cols)
                # without FHE
                col_indices = np.nonzero(cols)[0]
                cols2remove = col_indices[np.where(col_msr > self.multiple_node_deletion_threshold * msr)]
                cols[cols2remove] = False

                # with FHE
                """multiple_col_deletion = cols2remove()
                cols = multiple_col_deletion.multiple_cols_deletion.encrypt_run_decrypt(data.astype('uint16'),
                                                                                     col_msr.astype('uint16'),
                                                                                     msr.astype('uint16'))"""

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

            #without FHE
            cols2add = np.where(col_msr <= msr)[0]
            cols[cols2add] = True

            #With FHE
            """multiple_col_addition = cols2add()
            cols = multiple_col_addition.cols_addition.encrypt_run_decrypt(data.astype('uint16'))
"""
            msr, _, _ = self._calculate_msr(data, rows, cols)
            row_msr, row_inverse_msr = self._calculate_msr_row_addition(data, rows, cols)

            #Without FHE
            rows2add = np.where(np.logical_or(row_msr <= msr, row_inverse_msr <= msr))[0]
            rows[rows2add] = True

            #With FHE
            """multiple_row_addition = rowsaddition(data.astype('uint16'), row_msr.astype('uint16'), row_inverse_msr.astype('uint16'), msr.astype('uint16'))
            rows = multiple_row_addition.multiple_rows_addition.encrypt_run_decrypt(self)"""
            """multiple_row = rowsaddition()
            data = data.astype('uint16')
            row_msr = row_msr.astype('uint16')
            msr = msr.astype('uint16')
            rows = multiple_row.multiple_rows_deletion.encrypt_run_decrypt(data, row_msr, msr)"""

            if np.all(rows == rows_old) and np.all(cols == cols_old):
                stop = True

    def _calculate_msr(self, data, rows, cols):
        """Calculate the mean squared residues of the rows, of the columns and of the full data matrix."""

        #With FHE
        sample = data[rows][:, cols]
        sample = sample.astype('uint16')

        msrcalculator = EncryptedMsrCalculator()
        squared_residues = msrcalculator.squared_residues.encrypt_run_decrypt(sample)
        msr = msrcalculator.msr_calculator.encrypt_run_decrypt(squared_residues)
        row_msr = msrcalculator.row_msr_calculator.encrypt_run_decrypt(squared_residues)
        col_msr = msrcalculator.column_msr_calculator.encrypt_run_decrypt(squared_residues)

        #Without HE
        """   sub_data = data[rows][:, cols]
        data_mean = np.mean(sub_data)
        row_means = np.mean(sub_data, axis=1)
        col_means = np.mean(sub_data, axis=0)
        row_means = row_means[:, np.newaxis]
        residues = sub_data - row_means[:, np.newaxis] - col_means + data_mean
        squared_residues = residues * residues

        msr = np.mean(squared_residues)
        row_msr = np.mean(squared_residues, axis=1)
        col_msr = np.mean(squared_residues, axis=0)"""

        return msr, row_msr, col_msr

    def _calculate_msr_col_addition(self, data, rows, cols):
        """Calculate the mean squared residues of the columns for the node addition step."""

        #With FHE
        """"sample = data[rows][:, cols]
        sample = sample.astype('uint16')

        sample_rows = data[rows]
        sample_rows = sample_rows.astype('uint16')

        msrcolcalculator = EncryptedMsrColAdditionCalculator()
        squared_residues = msrcolcalculator.squared_residues.encrypt_run_decrypt(sample, sample_rows)
        col_msr = msrcolcalculator.msr_column_addition_calculator.encrypt_run_decrypt(squared_residues)"""

        #Without FHE
        sub_data = data[rows][:, cols]
        sub_data_rows = data[rows]

        data_mean = np.mean(sub_data)
        row_means = np.mean(sub_data, axis=1)
        col_means = np.mean(sub_data_rows, axis=0)

        col_residues = sub_data_rows - row_means[:, np.newaxis] - col_means + data_mean
        col_squared_residues = col_residues * col_residues
        col_msr = np.mean(col_squared_residues, axis=0)

        return col_msr

    def _calculate_msr_row_addition(self, data, rows, cols):
        """Calculate the mean squared residues of the rows and of the inverse of the rows for
        the node addition step."""

        #With FHE
        """sample = data[rows][:, cols]
        sample = sample.astype('uint16')

        sample_cols = data[:, cols]
        sample_cols = sample_cols.astype('uint16')

        msrrowcalculator = EncryptedMsrRowAdditionCalculator()
        row_squared_residues = msrrowcalculator.squared_residues_rows.encrypt_run_decrypt(sample, sample_cols)
        row_msr = msrrowcalculator.row_msr_calculator.encrypt_run_decrypt(row_squared_residues)
        inverse_squared_residues = msrrowcalculator.squared_residues_inverse_rows.encrypt_run_decrypt(sample, sample_cols)
        row_inverse_msr = msrrowcalculator.msr_inverse_calculator.encrypt_run_decrypt(inverse_squared_residues)"""

        #Without FHE
        sub_data = data[rows][:, cols]
        sub_data_cols = data[:, cols]
        data_mean = np.mean(sub_data)
        row_means = np.mean(sub_data_cols, axis=1)
        col_means = np.mean(sub_data, axis=0)

        row_residues = sub_data_cols - row_means[:, np.newaxis] - col_means + data_mean
        row_squared_residues = row_residues * row_residues
        row_msr = np.mean(row_squared_residues, axis=1)

        inverse_residues = -sub_data_cols + row_means[:, np.newaxis] - col_means + data_mean
        row_inverse_squared_residues = inverse_residues * inverse_residues
        row_inverse_msr = np.mean(row_inverse_squared_residues, axis=1)

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

