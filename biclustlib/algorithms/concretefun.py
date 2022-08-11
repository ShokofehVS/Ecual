import concrete.numpy as hnp
import numpy as np


class concretefun:
    def __init__(self, input_data):
        self.data = input_data

    def mean(self):
        np.sum(self.data).astype(np.uint16) // self.data.size

    def compile_mean(self):
        compiler = hnp.NPFHECompiler(self.mean, {"x": "encrypted"})
        inputset = [np.random.randint(0,600, size=(2884,17), dtype=np.uint16) for _ in range(10)]
        circuit = compiler.compile_on_inputset(inputset)
        circuit.keygen()
        public_args = circuit.encrypt(self.data)
        encrypted_result = circuit.run(public_args)
        return encrypted_result











