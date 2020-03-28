import sys
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from tensorflow.python.saved_model import tag_constants
import utils


def create_network(filename,k):
    # TODO check again the input op
    output_op_name = "model/pi/add"   # OUTPUT =  -11.130481
    # output_op_name = "model/concat"
    # output_op_name = "model/split"
    # output_op_name = "model/add"
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



    outputVars = network.outputVars
    print("outputVars =", outputVars)
    print("outputVars len =", len(outputVars))

    assert (len(outputVars) == k)
    print("network outputVars =", network.outputVars)

    # epsilon for bounding latency gradient (for each k)
    latency_gradient_eps = []
    for i in range(k):
        eps = network.getNewVariable()
        print()
        network.setLowerBound(eps, -0.01)
        network.setUpperBound(eps, 0.01)
        latency_gradient_eps.append(eps)

    # epsilon for bounding latency ratio
    latency_ratio_eps = network.getNewVariable()
    network.userDefineInputVars.append(latency_ratio_eps)

    network.setLowerBound(latency_ratio_eps, 1)
    network.setUpperBound(latency_ratio_eps, 1.01)

    # # epsilon for separating inputs
    # new_inputs_eps = []
    # for i in range(k):
    #     eps = network.getNewVariable()
    #     network.setLowerBound(eps, 1/1e2)
    #     network.setUpperBound(eps, 1e9)
    #     new_inputs_eps.append(eps)

    # latency gradient new inputs
    new_inputs = []
    for i in range(k):
        new_intput = network.getNewVariable()
        network.userDefineInputVars.append(new_intput)

        print("new input var = ", new_intput)
        new_inputs.append(new_intput)
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
        eq.addAddend(-1, latency_gradient_eps[i])
        eq.addAddend(1, new_intput)
        eq.setScalar(0)
        network.addEquation(eq)

    b = 0
    for j in range(k):
        inputVars = network.inputVars[j][0]
        latency_gradient_indices = [i for i in range(0, len(inputVars), 3)]
        latency_ratio_indices = [i + 1 for i in range(0, len(inputVars), 3)]
        sending_ratio_indices = [i + 2 for i in range(0, len(inputVars), 3)]
        # print("latency_gradient_indices:")
        for i in latency_gradient_indices:
            # print("i = ",i,"inputVars[i]=",   inputVars[i])
            # l = 0 - eps
            # u = 0 + eps
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            new_input_idx = (b+j)%(k)
            eq.addAddend(-1, inputVars[i])
            eq.addAddend(1, new_inputs[new_input_idx])
            # print("var",inputVars[i], " = input"+str(new_input_idx), "var idx = ", new_inputs[new_input_idx] )
            eq.setScalar(0)
            network.addEquation(eq)
            b+=1

        # print("latency_ratio_indices")
        for i in latency_ratio_indices:
            # print("i = ", i, "inputVars[i]=", inputVars[i])
            # l = 1
            # u = 1 + eps
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            # network.userDefineInputVars.append(inputVars[i])
            eq.addAddend(1, inputVars[i])
            eq.addAddend(-1, latency_ratio_eps)
            eq.setScalar(0)
            network.addEquation(eq)

        # print("sending_ratio_indices")
        for i in sending_ratio_indices:
            # print("i = ", i, "inputVars[i]=", inputVars[i])
            l = 1 # No loss
            u = 1
            network.userDefineInputVars.append(inputVars[i])
            network.setUpperBound(inputVars[i], u)
            network.setLowerBound(inputVars[i], l)

    for i in range(len(outputVars)):
        network.setLowerBound(outputVars[i], -0.001)  # FOR K-query - 0.01 (), for 1 - query, any example with minus will do.
        network.setUpperBound(outputVars[i], 0.001)

    query_info = "-0.02<=latency_gradient<= 0.01, 1<=latency_ratio_indices<=1.01, sending_ratio_indices = 1\n" \
                 "output=0 (with a little error)"
    # return
    print("\nMarabou results:\n")
    # network.saveQuery("/cs/usr/tomerel/unsafe/VerifyingDeepRL/WP/proj/results/basic_query")
    # Call to C++ Marabou solver
    # if to_log_file:
    #     vals, stats = network.solve("results/vrl_marabou.log",verbose=False)
    #     print('marabou solve run result: {} '.format(
    #         'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))
    # else:
    #     vals, stats = network.solve(verbose=True)
    #     print(vals)
    #     print('marabou solve run result: {} '.format(
    #         'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))

    vals, stats = network.solve()#"results/vrl_marabou.log",verbose=False)#, options = options)
    print(vals)
    result = 'SAT' if len(list(vals.items())) != 0 else 'UNSAT'
    print('marabou solve run result: {} '.format(
        result))
    # TODO: fix inputs
    # utils.write_results_to_file(vals,inputVars, outputVars, "K-query1",query_info,".",k)
    return result
    # utils.write_results_to_file(vals,inputVars, outputVars, "K-query1",query_info,".",k)


import sys

def main():

    if len(sys.argv) not in [3]:
        print("usage:",sys.argv[0], "<pb_filename> [k] ")
        exit(0)
    filename = sys.argv[1]
    k = int(sys.argv[2])
    k_test(filename,k,False)

if __name__ == "__main__":
    main()
