import concrete.numpy as cnp
import numpy as np

class SingleNodeDeletion:
    single_node_deletion_circuit: cnp.Circuit
    comparison_circuit: cnp.Circuit

    def __init__(self, inputset):
        @cnp.compiler({"msr": "encrypted"})
        def single_node_deletion_calculator(msr):
            while msr > 300:
                return True

        self.single_node_deletion_circuit = single_node_deletion_calculator.compile(inputset)

        @cnp.compiler({"msr": "encrypted"})
        def comparison(msr):
            return cnp.univariate(single_node_deletion_calculator)(msr)

        self.comparison_circuit = comparison.compile(inputset)
