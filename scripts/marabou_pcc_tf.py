
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
    # 0 - 9   : latency gradient, the derivative of latency with respect to time
    # 10 - 19 : latency ratio, the ratio of the current MI’s mean latency to minimum observed mean latency of any MI in
    #           the connection’s history
    # 20 - 29 : sending ratio, the ratio of packets sent to packets acknowledged by the receiver
    for i in range (0,10):
        network.setLowerBound(inputVars[i],0)
        network.setUpperBound(inputVars[i], 0)
    for i in range(10, 20):
        network.setLowerBound(inputVars[i], 0)
        network.setUpperBound(inputVars[i], 1)
    for i in range(20, 30):
        network.setLowerBound(inputVars[i], 0)
        network.setUpperBound(inputVars[i], 1e9)


    # Set output bounds
    for i in range(len(outputVars)):
        network.setLowerBound(outputVars[i], 0)
        network.setUpperBound(outputVars[i], 0)
    print("\n===== Marabou =====\n")
    # Call to C++ Marabou solver
    vals, stats = network.solve("results/vrl_marabou.log")



import sys

def main():
    if len(sys.argv)!=2:
        print("usage:",sys.argv[0], "<pb_filename>")
        exit(0)
    filename = sys.argv[1]
    run_marabou(filename)


if __name__ == "__main__":
    main()
