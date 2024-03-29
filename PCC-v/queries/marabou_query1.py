
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from eval_network import evaluateNetwork
from tensorflow.python.saved_model import tag_constants



def create_network(filename):
    output_op_name = "output/Shape"
    input_op_names = ["input/Ob"]
    # read_tf(filename, inputName=None, outputName=None, savedModel=False, savedModelTags=[]):
    sanity_inputs = [0. ,        1.,         1.06896552 ,0. ,        1.  ,       1.01694915,
     0.,         1.,         1.01694915, 0.,         1. ,        1.03508772,
     0.,         1.,         1.09090909 ,0.,         1.,         1.07142857,
     0.,         1.,         1.03389831, 0.,         1.,         1.,
     0. ,        1. ,        1.01666667 ,0.,         1. ,        1.07017544]



    print("my inputs:", sanity_inputs)

    sanity_inputs = np.asanyarray(sanity_inputs).reshape ((1,30))
    print( "EXPECTED OUTPUT IS ONE OF :  [0.035300, 0.004130,  -0.001897] ")
    print("network output for my inputs(MY):", evaluateNetwork(filename, sanity_inputs, input_op_names, output_op_name))
    exit (0)
    network = Marabou.read_tf(filename, inputName=input_op_names,outputName=output_op_name) #,savedModel = True,outputName = "save_1/restore_all", savedModelTags=[tag_constants.SERVING] )
    return network, input_op_names, output_op_name

# Network inputs:
# 0 - 9   : latency gradient, the derivative of latency with respect to time
# 10 - 19 : latency ratio, the ratio of the current MI’s mean latency to minimum observed mean latency of any MI in
#           the connection’s history
# 20 - 29 : sending ratio, the ratio of packets sent to packets acknowledged by the receiver

def basic_test(filename, to_log_file):

    network,input_op_names, output_op_name =  create_network(filename)

    # Get the input and output variable numbers; [0] since first dimension is batch size
    inputVars = network.inputVars[0][0]

    outputVars = network.outputVars[0]
    print("inputVars len =", len(inputVars))
    print("outputVars len =", len(outputVars))
    print("outputVars =", outputVars)
    print("network outputVars =", network.outputVars)
    print("outputVars[0]  =", outputVars[0])
    print("outputVars[0].type  =", type(outputVars[0]))
    print(network.inputVars)

    sanity_inputs =[]
    eps0 = network.getNewVariable()
    eps1 = network.getNewVariable()

    network.setLowerBound(eps0, -0.01)
    network.setUpperBound(eps0, 0.01)
    network.setLowerBound(eps1, 0)
    network.setUpperBound(eps1, 0.01)
    for i in range (0,10):
        # l = 0 - eps
        # u = 0 + eps

        # network.setUpperBound(inputVars[i], u)
        # network.setLowerBound(inputVars[i],l)
        sanity_inputs.append(0)
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
        eq.addAddend(1, eps0)
        eq.addAddend(-1, inputVars[i])
        eq.setScalar(0)
        network.addEquation(eq)

    for i in range(10, 20):
        # l = 1
        # u = 1 + eps

        # network.setUpperBound(inputVars[i], u)
        # network.setLowerBound(inputVars[i], l)
        sanity_inputs.append(1)
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
        eq.addAddend(1, inputVars[i])
        eq.addAddend(-1, eps1)
        eq.setScalar(1)
        network.addEquation(eq)

    for i in range(20, 30):
        l = 1
        u = 1
        network.setUpperBound(inputVars[i], u)
        network.setLowerBound(inputVars[i], l)
        sanity_inputs.append((u+l)//2)

    for i in range(len(outputVars)):
        network.setLowerBound(outputVars[i], -100) # as any minus will cause wrong behavior (will force, eventually
        # converting to the MIN RATE)
        network.setUpperBound(outputVars[i], 0)

    sanity_inputs = [0. ,        1.,         1.06896552 ,0. ,        1.  ,       1.01694915,
     0.,         1.,         1.01694915, 0.,         1. ,        1.03508772,
     0.,         1.,         1.09090909 ,0.,         1.,         1.07142857,
     0.,         1.,         1.03389831, 0.,         1.,         1.,
     0. ,        1. ,        1.01666667 ,0.,         1. ,        1.07017544]

    print( "EXPECTED OUTPUT IS ONE OF :  [0.035300, 0.004130,  -0.001897] ")

    sanity_inputs = np.asanyarray(sanity_inputs).reshape ((1,30))

    print("my inputs:", sanity_inputs)

    print("network output for my inputs(MY):", evaluateNetwork(filename, sanity_inputs, input_op_names, output_op_name))
    # print ("network output for my inputs(func BY MARABOU):",network.My_evaluateWithoutMarabou([sanity_inputs],output_op_name))

    print("\nMarabou results:\n")


    # network.saveQuery("/cs/usr/tomerel/unsafe/VerifyingDeepRL/WP/proj/results/basic_query")
    # Call to C++ Marabou solver
    if to_log_file:
        vals, stats = network.solve("results/vrl_marabou.log",verbose=False)
        print('marabou solve run result: {} '.format(
            'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))
    else:
        vals, stats = network.solve(verbose=True)
        print(vals)
        print('marabou solve run result: {} '.format(
            'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))





import sys

def main():

    if len(sys.argv) not in [2,3]:
        print("usage:",sys.argv[0], "<pb_filename> [-l]")
        exit(0)

    filename = sys.argv[1]
    print("=========================-basic_test-=========================")
    basic_test(filename, len(sys.argv) == 3)



if __name__ == "__main__":
    main()
