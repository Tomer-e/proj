import sys
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from eval_network import evaluateNetwork
from tensorflow.python.saved_model import tag_constants

## SD QUERY : the situation is great and we were expected bitrate to be HD, but actual bitrate is SD

def create_network(filename,k):
    # TODO check again the input op

    input_op_names = ["Placeholder", "Placeholder_1", "Placeholder_2", "Placeholder_3", "Placeholder_4", "Placeholder_5", "Placeholder_6", "Placeholder_7", "Placeholder_8", "Placeholder_9", "Placeholder_10", "Placeholder_11", "Placeholder_12", "Placeholder_13", "Placeholder_14", "Placeholder_15", "Placeholder_16", "Placeholder_17", "Placeholder_18", "Placeholder_19", "Placeholder_20", "Placeholder_21", "Placeholder_22", "Placeholder_23", "Placeholder_24", "Placeholder_25", "Placeholder_26", "Placeholder_27", "Placeholder_28", "Placeholder_29", "Placeholder_30", "Placeholder_31", "Placeholder_32", "Placeholder_33", "Placeholder_34", "Placeholder_35", "Placeholder_36", "Placeholder_37", "Placeholder_38", "Placeholder_39", "Placeholder_40"]

    output_op_name = "actor_agent_30/fully_connected_7/BiasAdd"

    network = Marabou.read_tf(filename,inputName=input_op_names,outputName=output_op_name)
    return network, input_op_names, output_op_name


# Network inputs:
# TODO
def k_test(filename,k):
    network, input_op_names, output_op_name = create_network(filename,k)
    inputVars = network.inputVars
    outputVars = network.outputVars
    print ("inputVars:", inputVars)
    print ("outputVars:", outputVars)
    # print ("len outputVars:",len(outputVars))

    for j in range(len(outputVars)):
        network.setLowerBound(outputVars[j], -1e6)
        network.setUpperBound(outputVars[j], 1e6)


    # choose k for the last chunk
    # eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
    #
    # # The right one:
    # # eq.addAddend(1, outputVars[(utils.S_LEN - 1) * utils.A_DIM])
    # # eq.addAddend(-1, outputVars[-1]) # outputVars[utils.S_LEN - 1) * utils.A_DIM + (utils.A_DIM - 1)]
    #
    # # The sanity one:
    # eq.addAddend(1, outputVars[-1])  # outputVars[utils.S_LEN - 1) * utils.A_DIM + (utils.A_DIM - 1)]
    # eq.addAddend(-1, outputVars[(utils.S_LEN - 1) * utils.A_DIM])
    # eq.setScalar(0)
    # network.addEquation(eq)


    print("\nMarabou results:\n")

    # network.saveQuery("/cs/usr/tomerel/unsafe/VerifyingDeepRL/WP/proj/results/basic_query")
    # Call to C++ Marabou solver
    vals, stats = network.solve(verbose=True)
    print(vals)
    print('marabou solve run result: {} '.format(
        'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))
    return result


def main():

    if len(sys.argv) not in [3]:
        print("usage:",sys.argv[0], "<pb_filename> [k] ")
        exit(0)
    filename = sys.argv[1]
    k = int(sys.argv[2])
    k_test(filename,k)

if __name__ == "__main__":
    main()
