import sys
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from eval_network import evaluateNetwork
from tensorflow.python.saved_model import tag_constants
import utils


def create_network(filename,k):
    # TODO check again the input op
    output_op_name = "model/pi/add"   # OUTPUT =  -11.130481
    # output_op_name = "model/concat"   #  NODE 177 -11.130481
    # output_op_name = "model/split"    #  NODE 176 -11.130481
    # output_op_name = "model/add"        #  NODE 176 -11.130481
    input_op_names = ["input/Ob"]
    # read_tf(filename, inputName=None, outputName=None, savedModel=False, savedModelTags=[]):
    network = Marabou.read_tf_k_steps(filename, k,inputName=input_op_names,outputName=output_op_name) #,savedModel = True,outputName = "save_1/restore_all", savedModelTags=[tag_constants.SERVING] )
    return network, input_op_names, output_op_name

# Network inputs:
# 0 - 9   : latency gradient, the derivative of latency with respect to time
# 10 - 19 : latency ratio, the ratio of the current MI’s mean latency to minimum observed mean latency of any MI in
#           the connection’s history
# 20 - 29 : sending ratio, the ratio of packets sent to packets acknowledged by the receiver

def k_test(filename,k, to_log_file=False):
    network, input_op_names, output_op_name = create_network(filename,k)

    # Loss is pretty big, latency is ok +-
    outputVars = network.outputVars
    print("outputVars =", outputVars)
    print("outputVars len =", len(outputVars))

    assert (len(outputVars) == k)
    print("network outputVars =", network.outputVars)

    # epsilon for bounding latency gradient
    latency_gradient_eps = network.getNewVariable()
    network.setLowerBound(latency_gradient_eps, -0.02)
    network.setUpperBound(latency_gradient_eps, 0.02)

    # epsilon for bounding latency ratio
    latency_ratio_eps = network.getNewVariable()
    network.setLowerBound(latency_ratio_eps, 1)
    network.setUpperBound(latency_ratio_eps, 1.02)

    # epsilon for bounding sending ratio
    sending_ratio_eps = []
    for i in range(k):
        eps = network.getNewVariable()
        # network.userDefineInputVars.append(eps)
        network.setLowerBound(eps, 1.05)
        network.setUpperBound(eps, 100)
        sending_ratio_eps.append(eps)

    # sending ratio new inputs
    new_inputs = []
    for i in range(k):
        new_intput = network.getNewVariable()
        network.userDefineInputVars.append(new_intput)

        print("new input var = ", new_intput)
        new_inputs.append(new_intput)
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
        eq.addAddend(-1, sending_ratio_eps[i])
        eq.addAddend(1, new_intput)
        eq.setScalar(0)
        network.addEquation(eq)


    print ("---------------------",new_inputs)
    # # epsilon for separating inputs
    # new_inputs_eps = []
    # for i in range(k):
    #     eps = network.getNewVariable()
    #     network.setLowerBound(eps, 1/1e2)
    #     network.setUpperBound(eps, 1e9)
    #     new_inputs_eps.append(eps)


    for j in range(k):
        inputVars = network.inputVars[j][0]
        latency_gradient_indices = [i for i in range(0, len(inputVars), 3)]
        latency_ratio_indices = [i + 1 for i in range(0, len(inputVars), 3)]
        sending_ratio_indices = [i + 2 for i in range(0, len(inputVars), 3)]

        for i in latency_gradient_indices:
            # l = 0 - eps
            # u = 0 + eps
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            # network.userDefineInputVars.append(inputVars[i])
            eq.addAddend(1, inputVars[i])
            eq.addAddend(-1, latency_gradient_eps)
            eq.setScalar(0)
            network.addEquation(eq)

        for i in latency_ratio_indices:
            # l = 1
            # u = 1 + eps
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            eq.addAddend(1, inputVars[i])
            # network.userDefineInputVars.append(inputVars[i])
            eq.addAddend(-1, latency_ratio_eps)
            eq.setScalar(0)
            network.addEquation(eq)

        # Loss is pretty big
        b = 0
        for i in sending_ratio_indices:
            # l = 1.05
            # u = 15
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            new_input_idx = (b+j)%(k)
            print ("i=",i)
            print ("j=",j)
            eq.addAddend(-1, inputVars[i])
            eq.addAddend(1, new_inputs[new_input_idx])
            print("var",inputVars[i], " = input" +str(new_input_idx), "var idx = ", new_inputs[new_input_idx] )
            eq.setScalar(0)
            network.addEquation(eq)
            b+=1

    for i in range(len(outputVars)):
        network.setLowerBound(outputVars[i], -0.05)
        network.setUpperBound(outputVars[i], 1000)

    query_info = "-0.02<=latency_gradient<= 0.02, 1<=latency_ratio_indices<=1.02, sending_ratio_indices = 1\n" \
                 "output>=0"

    # print(network.inputVars)
    # inputVars = np.array()


    # for j in range (1,k):
    #     # print(network.inputVars[j])
    #     inputVars = np.concatenate(inputVars,network.inputVars[j])

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
    # TODO: fix inputs
    utils.write_results_to_file(vals,[0], outputVars, "K-query3",query_info,".",k)


import sys

def main():

    if len(sys.argv) not in [2,3]:
        print("usage:",sys.argv[0], "<pb_filename> [k] ")
        exit(0)
    filename = sys.argv[1]
    if (len(sys.argv) == 2):
        k_test(filename,1,False)
    if (len(sys.argv) == 3):
        k = int(sys.argv[2])
        k_test(filename,k,False)

if __name__ == "__main__":
    main()
