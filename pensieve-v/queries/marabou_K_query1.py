import sys
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
import utils
from eval_network import evaluateNetwork
from tensorflow.python.saved_model import tag_constants



def create_network(filename,k):
    # TODO check again the input op
    input_op_names = ["actor/InputData/X"]
    output_op_name = "actor/FullyConnected_4/BiasAdd"

    network = Marabou.read_tf_k_steps(filename, k,inputName=input_op_names,outputName=output_op_name)
    return network, input_op_names, output_op_name

# Network inputs:
# x~t is the network throughput measurements for the past k video chunks;
# τ~t is the download time of the past k video chunks, which represents the time interval of the throughput measurements;
# n~t is a vector of m available sizes for the next video chunk;
# b~t is the current buffer level;
# c~t is the number of chunks remaining in the video;
# l~t is the bitrate at which the last chunk was downloaded.

def k_test(filename,k, to_log_file=False):
    network, input_op_names, output_op_name = create_network(filename,k)
    inputVars = network.inputVars
    print("inputVars" ,inputVars)
    print("inputVars[0][0]" ,inputVars[0][0])

    outputVars = network.outputVars
    print("outputVars =", outputVars)
    print("outputVars len =", len(outputVars))

    all_inputs, used_inputs, unused_inputs, last_chunk_bit_rate, current_buffer_size, past_chunk_throughput, \
    past_chunk_download_time, next_chunk_sizes, number_of_chunks_left = utils.prep_input_for_query(inputVars, k)



    # # epsilon for bounding latency ratio
    # latency_ratio_eps = network.getNewVariable()
    # network.setLowerBound(latency_ratio_eps, 1)
    # network.setUpperBound(latency_ratio_eps, 1.02)
    #
    # # epsilon for bounding sending ratio
    # sending_ratio_eps = []
    # for i in range(k):
    #     eps = network.getNewVariable()
    #     # network.userDefineInputVars.append(eps)
    #     network.setLowerBound(eps, 1.05)
    #     network.setUpperBound(eps, 20)
    #     sending_ratio_eps.append(eps)
    # for i in latency_gradient_indices:
    #     # l = 0 - eps
    #     # u = 0 + eps
    #     eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
    #     new_input_idx = (b + j) % (k)
    #     eq.addAddend(-1, inputVars[i])
    #     eq.addAddend(1, new_inputs[new_input_idx])
    #     print("var", inputVars[i], " = input" + str(new_input_idx), "var idx = ", new_inputs[new_input_idx])
    #     eq.setScalar(0)
    #     network.addEquation(eq)
    #     b += 1


    for var in unused_inputs:
        l = 0
        u = 0
        network.setLowerBound(var, l)
        network.setUpperBound(var, u)

    for i in range (k):

        # last_chunk_bit_rate
        for var in last_chunk_bit_rate[i]:
            l = 1
            u = 2
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

        # current_buffer_size
        for var in current_buffer_size[i]:
            l = 1
            u = 2
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

        # past_chunk_throughput
        for var in past_chunk_throughput[i]:
            l = 0.5
            u = 0.5
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

        # past_chunk_download_time
        for var in past_chunk_download_time[i]:
            l = 0.05
            u = 0.05
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

        # next_chunk_sizes
        for var in next_chunk_sizes[i]:
            l = 1
            u = 1000
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

        # number_of_chunks_left
        for var in number_of_chunks_left[i]:
            l = 1
            u = 1
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

    for i in range(len(outputVars)):
        network.setLowerBound(outputVars[i], -10000)
        network.setUpperBound(outputVars[i], 100000)

    # return
    print("\nMarabou results:\n")

    # network.saveQuery("/cs/usr/tomerel/unsafe/VerifyingDeepRL/WP/proj/results/basic_query")
    # Call to C++ Marabou solver
    if to_log_file:
        vals, stats = network.solve("results/vrl_marabou.log",verbose=False)
        print('marabou solve run result: {} '.format(
            'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))
    else:
        vals, stats = network.solve(verbose=True)
        # print(vals)
        print('marabou solve run result: {} '.format(
            'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))


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
