
from maraboupy import Marabou
import numpy as np
from tensorflow.python.saved_model import tag_constants



def run_marabou(filename):
    # read_tf(filename, inputName=None, outputName=None, savedModel=False, savedModelTags=[]):
    network = Marabou.read_tf(filename, inputName=["input/Ob"],outputName="model/split") #,savedModel = True,outputName = "save_1/restore_all", savedModelTags=[tag_constants.SERVING] )

    ## Or, you can specify the operation names of the input and output operations
    ## By default chooses the only placeholder as input, last op as output

    # Get the input and output variable numbers; [0] since first dimension is batch size
    inputVars = network.inputVars[0][0]

    outputVars = network.outputVars[0]
    print("inputVars len =", len(inputVars))
    print("outputVars len =", len(outputVars))
    print("outputVars[0]  =", outputVars[0])
    print("outputVars[0].type  =", type(outputVars[0]))
    # print("outputVars[0].shape  =", outputVars[0].shape)
    # exit(0)

    print(network.inputVars)

    # Set input bounds
    network.setLowerBound(inputVars[0],-10.0)
    network.setUpperBound(inputVars[0], 10.0)
    network.setLowerBound(inputVars[1],-10.0)
    network.setUpperBound(inputVars[1], 10.0)

    # Set output bounds
    network.setLowerBound(outputVars[0], 0)
    network.setUpperBound(outputVars[0], 210.0)
    print("\n===== Marabou =====\n")
    # Call to C++ Marabou solver
    vals, stats = network.solve() #("marabou.log")



import sys

def main():
    if len(sys.argv)!=2:
        print("usage:",sys.argv[0], "<pb_filename>")
        exit(0)
    filename = sys.argv[1]
    run_marabou(filename)


if __name__ == "__main__":
    main()
