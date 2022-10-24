import concrete.numpy as cnp
import numpy as np

class rowsaddition:

    def __init__(self, data, row_msr, inverse_msr, msr):
        self.data = data
        self.row_msr = row_msr
        self.inverse_msr = inverse_msr
        self.msr = msr

    @cnp.circuit({"data": "encrypted", "row_msr": "encrypted", "inverse_msr": "encrypted", "msr": "encrypted"}, verbose=True)
    def multiple_rows_addition(data: cnp.tensor[cnp.uint16, 2, 2], row_msr: cnp.tensor[cnp.uint16, 2, 2],
                               inverse_msr: cnp.tensor[cnp.uint16, 2, 2], msr: cnp.uint16):
        table = cnp.LookupTable([i for i in range(2 ** 6)] + [1 for _ in range(2 ** 6)])
        resulting_rows = []
        for i in range(data.shape[0]):

            row = data[i]
            condition = (row_msr <= msr) | (inverse_msr <= msr)
            # set the 7th bit of each value in the row to 0 or 1
            tagged_row = row + ((2 ** 6) * condition.shape[i])
            # depending on the 7th bit, set values to 0 or themselves
            rows2remove = table[tagged_row]
            resulting_rows.append(rows2remove)

        return np.concatenate(tuple(resulting_rows)).reshape(data.shape)

