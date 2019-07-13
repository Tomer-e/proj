
from maraboupy import Marabou
import numpy as np
from tensorflow.python.saved_model import tag_constants



def run_marabou(filename)
    network = Marabou.read_tf(filename) #,savedModel = True,outputName = "save_1/restore_all", savedModelTags=[tag_constants.SERVING] )

    ## Or, you can specify the operation names of the input and output operations
    ## By default chooses the only placeholder as input, last op as output

    # Get the input and output variable numbers; [0] since first dimension is batch size
    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars[0]

    # Set input bounds
    network.setLowerBound(inputVars[0],-10.0)
    network.setUpperBound(inputVars[0], 10.0)
    network.setLowerBound(inputVars[1],-10.0)
    network.setUpperBound(inputVars[1], 10.0)

    # Set output bounds
    network.setLowerBound(outputVars[1], 194.0)
    network.setUpperBound(outputVars[1], 210.0)

    # Call to C++ Marabou solver
    vals, stats = network.solve("marabou.log")



import sys

def main():
    if len(sys.argv)!=2:
        print("usage:",sys.argv[0], "<pb_filename>")
        exit(0)
    filename = sys.argv[1]
    run_marabou(filename)


if __name__ == "__main__":
    main()
