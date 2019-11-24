import sys
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from eval_network import evaluateNetwork
from tensorflow.python.saved_model import tag_constants



def create_network(filename,k):
    output_op_name = "model/pi/add"
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

    # Get the input and output variable numbers;
    inputVars = network.inputVars

    outputVars = network.outputVars
    print("inputVars len =", len(inputVars))
    print("inputVars:", network.inputVars)
    print("outputVars =", outputVars)
    print("outputVars len =", len(outputVars))
    assert (len(outputVars) == k)
    print("network outputVars =", network.outputVars)

    # epsilon for bounding latency gradient (for each i%k)
    latency_gradient_eps = []
    for i in range(k):
        eps = network.getNewVariable()
        network.setLowerBound(eps, -1)
        network.setUpperBound(eps, 1)
        latency_gradient_eps.append(eps)

    # epsilon for bounding latency ratio
    latency_ratio_eps = network.getNewVariable()

    network.setLowerBound(latency_ratio_eps, 0)
    network.setUpperBound(latency_ratio_eps, 1)

    # # epsilon for separate inputs
    # new_inputs_eps = []
    # for i in range(k):
    #     eps = network.getNewVariable()
    #     network.setLowerBound(eps, 1/1e2)
    #     network.setUpperBound(eps, 1e9)
    #     new_inputs_eps.append(eps)

    # latency gradient new inputs
    new_intpus = []
    for i in range(k):
        new_intput = network.getNewVariable()
        new_intpus.append(new_intput)
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
        eq.addAddend(1, new_intput)
        eq.addAddend(-1, latency_gradient_eps[i])
        eq.setScalar(0)
        network.addEquation(eq)

    # eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
    # eq.addAddend(1, a)
    # eq.addAddend(-1, b)
    # eq.addAddend(-1, eps2)
    # eq.setScalar(0)
    # network.addEquation(eq)

    for j in range(k):
        for i in range(0, 10):
            # l = 0 - eps
            # u = 0 + eps
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            eq.addAddend(1, inputVars[(30*j)+i])
            c_eps = (i+j)%(k)
            eq.addAddend(-1, new_intpus[c_eps])
            print("var",(30*j)+i, "will be like ",c_eps)
            eq.setScalar(0)
            network.addEquation(eq)

        for i in range(10, 20):
            # l = 1
            # u = 1 + eps
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            eq.addAddend(1, inputVars[(30*j)+i])
            eq.addAddend(-1, latency_ratio_eps)
            eq.setScalar(1)
            network.addEquation(eq)

        for i in range(20, 30):
            l = 1
            u = 1
            network.setUpperBound(inputVars[(30*j)+i], u)
            network.setLowerBound(inputVars[(30*j)+i], l)

    for i in range(len(outputVars)):
        network.setLowerBound(outputVars[i], -10)
        network.setUpperBound(outputVars[i], 10)

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


def basic_test(filename, to_log_file):

    network,input_op_names, output_op_name =  create_network(filename,2)

    # Get the input and output variable numbers;
    inputVars = network.inputVars

    outputVars = network.outputVars
    print("inputVars len =", len(inputVars))
    print("outputVars =", outputVars)
    print("outputVars len =", len(outputVars))
    print("network outputVars =", network.outputVars)
    print("outputVars[0]  =", outputVars[0])
    print("outputVars[0].type  =", type(outputVars[0]))
    print(network.inputVars)

    sanity_inputs =[]
    eps_a = network.getNewVariable()
    eps_b = network.getNewVariable()
    eps1 = network.getNewVariable()
    eps2 = network.getNewVariable()
    a = network.getNewVariable()
    b = network.getNewVariable()

    # epsilon for bounding latency gradient (when i%2==0)
    network.setLowerBound(eps_a, -15)
    network.setUpperBound(eps_a, 15)

    # epsilon for bounding latency gradient (when i%2==1)
    network.setLowerBound(eps_b, -15)
    network.setUpperBound(eps_b, 15)

    # epsilon for bounding latency ratio
    network.setLowerBound(eps1, 0)
    network.setUpperBound(eps1, 0.1)

    # epsilon for separate a and b
    network.setLowerBound(eps2, 1/1e2)
    network.setUpperBound(eps2, 1e9)

    eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
    eq.addAddend(1, a)
    eq.addAddend(-1, eps_a)
    eq.setScalar(0)
    network.addEquation(eq)

    eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
    eq.addAddend(1, b)
    eq.addAddend(-1, eps_b)
    eq.setScalar(0)
    network.addEquation(eq)

    eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
    eq.addAddend(1, a)
    eq.addAddend(-1, b)
    eq.addAddend(-1, eps2)
    eq.setScalar(0)
    network.addEquation(eq)


    for i in range (0, 10):
        # l = 0 - eps
        # u = 0 + eps

        # network.setUpperBound(inputVars[i], u)
        # network.setLowerBound(inputVars[i],l)
        sanity_inputs.append(0)
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
        eq.addAddend(1, inputVars[i])
        if i % 2 == 0:
            eq.addAddend(-1, a)
        else:
            eq.addAddend(-1, b)
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

    for i in range(30 + 0, 30 + 10):
        # l = 0 - eps
        # u = 0 + eps

        # network.setUpperBound(inputVars[i], u)
        # network.setLowerBound(inputVars[i],l)
        # sanity_inputs.append(0)
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
        eq.addAddend(1, inputVars[i])

        if i % 2 == 0:
            eq.addAddend(-1, b)
        else:
            eq.addAddend(-1, a)
        eq.setScalar(0)
        network.addEquation(eq)

    for i in range(30 + 10, 30 + 20):
        # l = 1
        # u = 1 + eps

        # network.setUpperBound(inputVars[i], u)
        # network.setLowerBound(inputVars[i], l)
        # sanity_inputs.append(1)
        eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
        eq.addAddend(1, inputVars[i])
        eq.addAddend(-1, eps1)
        eq.setScalar(1)
        network.addEquation(eq)

    for i in range(30 + 20, 30 + 30):
        l = 1
        u = 1
        network.setUpperBound(inputVars[i], u)
        network.setLowerBound(inputVars[i], l)
        # sanity_inputs.append((u + l) // 2)


    for i in range(len(outputVars)):
        network.setLowerBound(outputVars[i], -12)
        network.setUpperBound(outputVars[i], 12)

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
        print("usage:",sys.argv[0], "<pb_filename> [k] [-l] ")
        exit(0)
    filename = sys.argv[1]
    if (len(sys.argv) == 2):
        print("=========================-basic_test-=========================")
        basic_test(filename, False)
    if (len(sys.argv) == 3):
        k = int(sys.argv[2])
        k_test(filename,k,False)



if __name__ == "__main__":
    main()
