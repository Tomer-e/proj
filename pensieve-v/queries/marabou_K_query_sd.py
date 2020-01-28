import sys
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
import utils
from eval_network import evaluateNetwork
from tensorflow.python.saved_model import tag_constants

MARABOU_ERR = 0.001


DOWNLOAD_TIME = 0.1
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
    outputVars = network.outputVars
    # print ("outputVars:", outputVars)
    # print ("len outputVars:",len(outputVars))
    assert (len(outputVars)%utils.A_DIM  == 0)


    all_inputs, used_inputs, unused_inputs, last_chunk_bit_rate, current_buffer_size, past_chunk_throughput, \
    past_chunk_download_time, next_chunk_sizes, number_of_chunks_left = utils.prep_input_for_query(inputVars, k)

    past_chunk_download_time_eps = []
    for j in range(k):
        eps = network.getNewVariable()
        # network.userDefineInputVars.append(eps)
        # 0-4 SECONDS
        network.setLowerBound(eps, DOWNLOAD_TIME-MARABOU_ERR)
        network.setUpperBound(eps, DOWNLOAD_TIME+MARABOU_ERR)  # max : 4s
        past_chunk_download_time_eps.append(eps)

    # past_chunk_throughput_eps = []
    # for j in range(k):
    #     eps = network.getNewVariable()
    #     # network.userDefineInputVars.append(eps)
    #     network.setLowerBound(eps, 0.5)  # min throughput (for delay of 4s)
    #     network.setUpperBound(eps, 5)  #
    #     past_chunk_throughput_eps.append(eps)

    for var in unused_inputs:
        l = 0
        u = 0
        network.setLowerBound(var, l)
        network.setUpperBound(var, u)

    for j in range(k):

        # last_chunk_bit_rate
        # one of VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        for var in last_chunk_bit_rate[j]:
            l = utils.VIDEO_BIT_RATE[0]/utils.VIDEO_BIT_RATE[-1] -MARABOU_ERR# lowest definition // 300/4300
            u = utils.VIDEO_BIT_RATE[0]/utils.VIDEO_BIT_RATE[-1] +MARABOU_ERR# lowest definition // 300/43000
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)
            if j == 0:
                l = utils.VIDEO_BIT_RATE[1]/utils.VIDEO_BIT_RATE[-1] # default definition // 750/4300
                u = utils.VIDEO_BIT_RATE[1]/utils.VIDEO_BIT_RATE[-1] # default definition // 750/4300
                network.setLowerBound(var, l)
                network.setUpperBound(var, u)
            # else:
            #     eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
            #     eq.addAddend(1, var)
            #     eq.addAddend(-1, next_chunk_sizes[j-1][0]) # lowest quality from the previous time
            #     eq.setScalar(0)
            #     network.addEquation(eq)


        # current_buffer_size
        for var in current_buffer_size[j]:
            l = 0.4# +((4-DOWNLOAD_TIME*10)*(j))/10 - MARABOU_ERR # ((4-DOWNLOAD_TIME*10)*(j+1))/10 - MARABOU_ERR #
            u = 0.4# +((4-DOWNLOAD_TIME*10)*(j))/10 + MARABOU_ERR # ((4-DOWNLOAD_TIME*10)*(j+1))/10 + MARABOU_ERR #
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

        # past_chunk_throughput
        i = 0
        a = [0, 0, 0, 0, 0, 0, 0, 0]
        for var in past_chunk_throughput[j]:
            # l = ? //TODO
            # u = ? //TODO
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            eq.addAddend(1, var)
            if i>=(k-j):
                # eq.addAddend(1, past_chunk_throughput_eps[j])
                eq.addAddend(-0.1/DOWNLOAD_TIME, next_chunk_sizes[j-1][0])
            else:
                eq.addAddend(0, 0) # 0
            eq.setScalar(0)
            network.addEquation(eq)
            i+=1

        # past_chunk_download_time
        i=0
        a = [0,0,0,0,0,0,0,0]
        for var in past_chunk_download_time[j]:
            # l = 0.1
            # u = 40 => 4s
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            eq.addAddend(-1, var)
            if i>=(k-j)-1:
                eq.addAddend(1, past_chunk_download_time_eps[j])
                a [i] = 1
            else:
                eq.addAddend(0, 0) # 0
            eq.setScalar(0)
            network.addEquation(eq)
            i+=1
        print("past_chunk_download_time")
        print(a)

        # next_chunk_sizes
        size_i = 0
        # basic_size = 2 / utils.VIDEO_BIT_RATE[-1]  # =0.00046511627906976747 # 2 MB in HD, 4300 bps
        # sizes = [basic_size * bitrate for bitrate in utils.VIDEO_BIT_RATE]
        # chunk_size_lower_bounds = [.126289,.318564,.520315, .801058, 1.22426, 1.941625]
        # chunk_size_upper_bounds = [.181801,.450283,.709534, 1.060487, 1.728879, 2.354772]
        chunk_size_lower_bounds = [.1, .3, .5, .8,   1.2, 1.5]
        chunk_size_upper_bounds = [.3, .6, .9,  1.3, 2, 2.4]

        assert len(next_chunk_sizes[j]) == len(utils.VIDEO_BIT_RATE)
        for var in next_chunk_sizes[j]:
            # All sizes
            # chunk_size = utils.VIDEO_BIT_RATE[size_i]
            # print("chunk_size", chunk_size)
            l = chunk_size_lower_bounds[size_i]# sizes[size_i]  # chunk_size
            u = chunk_size_upper_bounds[size_i]# sizes[size_i]  # chunk_size
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)
            size_i += 1

        # number_of_chunks_left
        for var in number_of_chunks_left[j]:
            l = (k-j)-1 / (k)
            u = (k-j)-1 / (k)
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

    for j in range(len(outputVars)):
        network.setLowerBound(outputVars[j], -1e6)
        network.setUpperBound(outputVars[j], 1e6)


    # choose hd k-1 first chunks
    # for j in range (utils.S_LEN-1):
    #     for i in range(utils.A_DIM-1):
    #         eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
    #         eq.addAddend(1, outputVars[j*utils.A_DIM+(utils.A_DIM - 1)])
    #         eq.addAddend(-1, outputVars[j*utils.A_DIM+i])
    #         eq.setScalar(0)
    #         network.addEquation(eq)

    # choose k for the last chunk
    eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)

    # The right one:
    # eq.addAddend(1, outputVars[(utils.S_LEN - 1) * utils.A_DIM])
    # eq.addAddend(-1, outputVars[-1]) # outputVars[utils.S_LEN - 1) * utils.A_DIM + (utils.A_DIM - 1)]

    # The sanity one:
    eq.addAddend(1, outputVars[-1])  # outputVars[utils.S_LEN - 1) * utils.A_DIM + (utils.A_DIM - 1)]
    eq.addAddend(-1, outputVars[(utils.S_LEN - 1) * utils.A_DIM])
    eq.setScalar(0)
    network.addEquation(eq)


    print("\nMarabou results:\n")

    # network.saveQuery("/cs/usr/tomerel/unsafe/VerifyingDeepRL/WP/proj/results/basic_query")
    # Call to C++ Marabou solver
    if to_log_file:
        vals, stats = network.solve("results/vrl_marabou.log", verbose=False)
        print('marabou solve run result: {} '.format(
            'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))
    else:
        vals, stats = network.solve(verbose=True)
        # print(vals)
        print("all_inputs = ", all_inputs)
        print("used_inputs = ", used_inputs)

        result = 'SAT' if len(list(vals.items())) != 0 else 'UNSAT'
        print('marabou solve run result: {} '.format(result))

        if result == 'SAT':
            for j in range(k):
                print(j, "/", k)
                print("last_chunk_bit_rate:")
                for var in last_chunk_bit_rate[j]:
                    print("var", var, " = ", vals[var])

                print("current_buffer_size:")
                for var in current_buffer_size[j]:
                    print("var", var, " = ", vals[var])

                print("past_chunk_throughput:")
                for var in past_chunk_throughput[j]:
                    print("var", var, " = ", vals[var])

                print("past_chunk_download_time:")
                for var in past_chunk_download_time[j]:
                    print("var", var, " = ", vals[var])

                print("next_chunk_sizes:")
                for var in next_chunk_sizes[j]:
                    print("var", var, " = ", vals[var])

                print("number_of_chunks_left:")
                for var in number_of_chunks_left[j]:
                    print("var", var, " = ", vals[var])

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
