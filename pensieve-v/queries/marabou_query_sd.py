
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from eval_network import evaluateNetwork
from tensorflow.python.saved_model import tag_constants
import utils
import warnings
warnings.filterwarnings('ignore')




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
    for j in range(k):
        eps = network.getNewVariable()
        # network.userDefineInputVars.append(eps)
        # 0-4 SECONDS
        network.setLowerBound(eps, 0.4) # min : 4 sec
        network.setUpperBound(eps, 1.90) # max : 190 sec (=~4*48 sec)
        past_chunk_download_time_eps.append(eps)

    past_chunk_throughput_eps = []
    for j in range(k):
        eps = network.getNewVariable()
        # network.userDefineInputVars.append(eps)
        network.setLowerBound(eps, 0)
        network.setUpperBound(eps,0.03488372093023256 ) # max throughput (for delay of 4s)
        past_chunk_throughput_eps.append(eps)

    for var in unused_inputs:
        l = 0
        u = 0
        network.setLowerBound(var, l)
        network.setUpperBound(var, u)

    for j in range (k):

        # last_chunk_bit_rate
        # one of VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        for var in last_chunk_bit_rate[j]:
            l = utils.VIDEO_BIT_RATE[0]/utils.VIDEO_BIT_RATE[-1] # lowest definition // 300/4300
            u = utils.VIDEO_BIT_RATE[0]/utils.VIDEO_BIT_RATE[-1] # lowest definition // 300/4300
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

        # current_buffer_size
        # almost empty, less then one chunk
        for var in current_buffer_size[j]:
            l = 0    #
            u = 0.4  # 4 seconds
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

        # past_chunk_throughput
        for var in past_chunk_throughput[j]:
            # l = ? //TODO
            # u = ? //TODO
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            eq.addAddend(-1, var)
            eq.addAddend(1, past_chunk_throughput_eps[j])
            eq.setScalar(0)
            network.addEquation(eq)

        # past_chunk_download_time
        for var in past_chunk_download_time[j]:
            # l = 40 => 4s
            # u = 190 => 19s
            eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.EQ)
            eq.addAddend(-1, var)
            eq.addAddend(1, past_chunk_download_time_eps[j])
            eq.setScalar(0)
            network.addEquation(eq)

        # next_chunk_sizes
        size_i = 0
        basic_size = 2/4300 # =0.00046511627906976747 # 2 MB in 4300 bps
        sizes = [basic_size * bitrate for bitrate in utils.VIDEO_BIT_RATE]
        assert len (next_chunk_sizes[j]) == len (utils.VIDEO_BIT_RATE)
        for var in next_chunk_sizes[j]:
            # All sizes
            # chunk_size = utils.VIDEO_BIT_RATE[size_i]
            # print("chunk_size", chunk_size)
            l = sizes[size_i]  # chunk_size
            u = sizes[size_i]  # chunk_size
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)
            size_i +=1

        # number_of_chunks_left
        for var in number_of_chunks_left[j]:
            l = 1/48 # only one chunk left to play
            u = 1/48 # only one chunk left to play
            network.setLowerBound(var, l)
            network.setUpperBound(var, u)

    for j in range(len(outputVars)):
        network.setLowerBound(outputVars[j], -1e9)
        network.setUpperBound(outputVars[j], 1e9)

    # hd > sd
    # for i in range (len (outputVars)-1):
    eq = MarabouUtils.Equation(EquationType=MarabouCore.Equation.GE)
    eq.addAddend(1, outputVars[-2])
    eq.addAddend(-1, outputVars[0])
    eq.setScalar(0)
    network.addEquation(eq)

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
