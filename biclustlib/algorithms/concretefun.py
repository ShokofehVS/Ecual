import concrete.numpy as hnp
import numpy as np
import concrete.numpy as cnp

class concretefun:
    # def __init__(self, input_data1, input_data2):
    #     self.op1 = input_data1
    #     self.op2 = input_data2

    # def mean(self, input):
    #     np.sum(input, axis=0).astype(np.uint16) // input.shape[0]
    #
    # def compile_mean(self):
    #     compiler = hnp.NPFHECompiler(self.mean, {"x": "encrypted"})
    #     inputset = [np.random.randint(0,600, size=(2884,17), dtype=np.uint16) for _ in range(10)]
    #     circuit = compiler.compile_on_inputset(inputset)
    #     circuit.keygen()
    #     public_args = circuit.encrypt(self.data)
    #     encrypted_result = circuit.run(public_args)
    #     return encrypted_result

    # def sum_enc(self,):
    #     self.op1 + self.op2
    #
    # # def compile_sum(self,var1, var2):
    # #     compiler = hnp.NPFHECompiler(self.sum_enc, {"x": "encrypted", "y": "encrypted"})
    # #     var1 = np.asarray(var1)
    # #     print(var1)
    # #     print(var2)
    # #     inputset = [(var1, var2)]
    # #     circuit = compiler.compile_on_inputset(inputset)
    # #     print(compiler)
    # #     print(circuit.encrypt_run_decrypt([(var1.astype(dtype=np.uint16), var2.astype(dtype=np.uint16))]))
    # #     # circuit.keygen()
    # #     # public_args = circuit.encrypt([(var1.astype(dtype=np.uint16), var2.astype(dtype=np.uint16))])
    # #     # encrypted_result = circuit.run(public_args)
    # #     # return encrypted_result
    # def submatrix(self, data, rows, cols):
    #     subdata = data[rows][:, cols]
    #     return subdata

    # def compiling(self):
    @cnp.circuit({"squared_residues": "encrypted"}, verbose=True)
    def def_row_msr(squared_residues: cnp.tensor[cnp.uint16, 10, 5]):
        row_msr = np.sum(squared_residues, axis=1) // squared_residues.shape[1]
        return row_msr











