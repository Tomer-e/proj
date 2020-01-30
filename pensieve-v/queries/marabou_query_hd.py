
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from eval_network import evaluateNetwork
from tensorflow.python.saved_model import tag_constants
import utils
import warnings
warnings.filterwarnings('ignore')



MARABOU_ERR = 0.001

DOWNLOAD_TIME = 0.1
def create_network(filename):
    # output_op_name = "actor/FullyConnected_4/Softmax"
    # input_op_names = ["actor/InputData/X"]
    # input_op_names = ["actor/strided_slice/stack", "actor/strided_slice_1/stack","actor/strided_slice_2/stack"
    #     ,"actor/strided_slice_3/stack","actor/strided_slice_4/stack","actor/strided_slice_5/stack"]
    # output_op_name = "actor/FullyConnected/MatMul"


    input_op_names = ["actor/InputData/X"]#, "actor/strided_slice_1","actor/strided_slice_2","actor/strided_slice_3","actor/strided_slice_4","actor/strided_slice_5"]
    # output_op_name = "add"
    output_op_name = "actor/FullyConnected_4/BiasAdd"
    # output_op_name = "critic/FullyConnected_4/BiasAdd"

    network = Marabou.read_tf(filename, inputName=input_op_names,outputName=output_op_name)
    return network, input_op_names, output_op_name

# Network inputs:
# x~t is the network throughput measurements for the past k video chunks;
# τ~t is the download time of the past k video chunks, which represents the time interval of the throughput measurements;
# n~t is a vector of m available sizes for the next video chunk;
# b~t is the current buffer level;
# c~t is the number of chunks remaining in the video;
# l~t is the bitrate at which the last chunk was downloaded.

def basic_test(filename, to_log_file):
    k = 1
    network,input_op_names, output_op_name = create_network(filename)

    inputVars = network.inputVars
    outputVars = network.outputVars[0]

    all_inputs, used_inputs, unused_inputs, last_chunk_bit_rate, current_buffer_size, past_chunk_throughput, \
    past_chunk_download_time, next_chunk_sizes, number_of_chunks_left = utils.prep_input_for_query(inputVars, k)


    past_chunk_download_time_eps = []
    for j in range(utils.S_LEN):
        eps = network.getNewVariable()
        # network.userDefineInputVars.append(eps)
        # 0-4 SECONDS
        network.setLowerBound(eps, DOWNLOAD_TIME-MARABOU_ERR)
        network.setUpperBound(eps, DOWNLOAD_TIME+MARABOU_ERR)
        past_chunk_download_time_eps.append(eps)

    chunk_size_eps = []
    for j in range(utils.S_LEN):
        eps = network.getNewVariable()
        # network.userDefineInputVars.append(eps)
        network.setLowerBound(eps, 1.9) #
        network.setUpperBound(eps, 2.5) #
        chunk_size_eps.append(eps)

    for var in unused_inputs:
        l = 0
        u = 0
        network.setLowerBound(var, l)
        network.setUpperBound(var, u)

    for j in range (k):

        # last_chunk_bit_rate
        # one of VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        for var in last_chunk_bit_rate[j]:
            l =1-0.05 # Highest definition // 4300/4300
            u =1+MARABOU_ERR # Highest definition //
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

        # current_buffer_size
        for var in current_buffer_size[j]:
            l = 0.8
            u = 19
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

        # past_chunk_throughput
        i = 0
        for var in past_chunk_throughput[j]:
            # l = 0 #//TODO
            # u = 10000 #//TODO
            # network.setLowerBound(var, l)
            # network.setUpperBound(var, u)
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            eq.addAddend(1, var)
            eq.addAddend(-0.1/DOWNLOAD_TIME, chunk_size_eps[i])
            # eq.addAddend(1, chunk_size_eps[i])
            eq.setScalar(0)
            network.addEquation(eq)
            i+=1

        # past_chunk_download_time
        i = 0
        for var in past_chunk_download_time[j]:
            # l = 0.1
            # u = 40 => 4s
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            eq.addAddend(-1, var)
            eq.addAddend(1, past_chunk_download_time_eps[j])
            eq.setScalar(0)
            network.addEquation(eq)
            i+=1


        # next_chunk_sizes
        # basic_size = 2/utils.VIDEO_BIT_RATE[-1] # =0.00046511627906976747 # 2 MB in HD, 4300 bps
        # sizes = [basic_size * bitrate for bitrate in utils.VIDEO_BIT_RATE]
        size_i = 0
        # chunk_size_lower_bounds = [.1, .3, .5, .8,   1.2, 1.5]
        # chunk_size_upper_bounds = [.3, .6, .9,  1.3, 2, 2.4]

        chunk_size_lower_bounds =[.126289, .318564, .520315, .801058, 1.22426, 1.941625]
        chunk_size_upper_bounds =[.181801, .450283, .709534, 1.060487, 1.728879, 2.354772]
        assert len (next_chunk_sizes[j]) == len (utils.VIDEO_BIT_RATE)
        for var in next_chunk_sizes[j]:
            # All sizes
            # chunk_size = utils.VIDEO_BIT_RATE[size_i]
            # print("chunk_size", chunk_size)
            l = chunk_size_lower_bounds[size_i]  # chunk_size
            u = chunk_size_upper_bounds[size_i]  # chunk_size
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)
            size_i +=1

        # number_of_chunks_left
        for var in number_of_chunks_left[j]:
            l = 0 - MARABOU_ERR# 0/48 # only one chunk left to play
            u = 0 + MARABOU_ERR# 0/48 # only one chunk left to play
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

    for j in range(len(outputVars)):
        network.setLowerBound(outputVars[j], -1e9)
        network.setUpperBound(outputVars[j], 1e9)

    # # SD > HD
    eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
    eq.addAddend(1, outputVars[0])
    eq.addAddend(-1, outputVars[5])
    eq.setScalar(0)
    network.addEquation(eq)

    print("\nMarabou results:\n")

    # network.saveQuery("/cs/usr/tomerel/unsafe/VerifyingDeepRL/WP/proj/results/basic_query")
    # Call to C++ Marabou solver
    # if to_log_file:
    #     vals, stats = network.solve("results/vrl_marabou.log",verbose=False)
    #     print('marabou solve run result: {} '.format(
    #         'SAT' if len(list(vals.items())) != 0 else 'UNSAT'))
    # else:
    vals, stats = network.solve(verbose=True)
    # print(vals)
    print("all_inputs = ",all_inputs)
    print("used_inputs = ",used_inputs)

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
        print("usage:",sys.argv[0], "<pb_filename> [-l]")
        exit(0)

    filename = sys.argv[1]
    print("=========================-basic_test-=========================")
    basic_test(filename, len(sys.argv) == 3)







if __name__ == "__main__":
    main()
