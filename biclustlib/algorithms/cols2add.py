import concrete.numpy as cnp
import numpy as np

class cols2add:

    @cnp.circuit({"data": "encrypted", "col_msr": "encrypted", "msr": "encrypted"}, verbose=True)
    def cols_addition(data: cnp.tensor[cnp.uint16, 2, 2], col_msr: cnp.tensor[cnp.uint16, 2, 2], msr: cnp.uint16):
        table = cnp.LookupTable([i for i in range(2 ** 6)] + [0 for _ in range(2 ** 6)])
        resulting_cols = []
        for i in range(data.shape[1]):
            col = data[i]
            condition = (col_msr <= msr)
            # set the 7th bit of each value in the row to 0 or 1
            tagged_col = col + ((2 ** 6) * condition.shape[i])
            # depending on the 7th bit, set values to 0 or themselves
            cols2add = table[tagged_col]
            resulting_cols.append(cols2add)

        return np.concatenate(tuple(resulting_cols)).reshape(data.shape)

