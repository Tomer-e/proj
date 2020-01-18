
from maraboupy import Marabou, MarabouUtils, MarabouCore
import numpy as np
from eval_network import evaluateNetwork
from tensorflow.python.saved_model import tag_constants
import warnings
warnings.filterwarnings('ignore')

M = A_DIM = 6
S_INFO = 6
K = S_LEN = 8


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

    network,input_op_names, output_op_name =  create_network(filename)

    # Get the input and output variable numbers; [0] since first dimension is batch size
    inputVars = network.inputVars[0]

    outputVars = network.outputVars[0]
    print("inputVars len =", len(inputVars))
    print("network.inputVars", network.inputVars)
    print("outputVars len =", len(outputVars))
    print("outputVars =", outputVars)
    print("network outputVars =", network.outputVars)
    print("outputVars[0]  =", outputVars[0])
    print("outputVars[0].type  =", type(outputVars[0]))
    print(network.inputVars)

    print ("inputVrs shape = ", inputVars.shape)
    all_inputs = set(inputVars.flatten())
    used_inputs  = set()
    last_chunk_bit_rate = inputVars[:, 0:1, -1] [0]      #[1]  l~t
    assert (len(last_chunk_bit_rate) == 1)
    for i in last_chunk_bit_rate:
        used_inputs.add(i)
        print(i)

    current_buffer_size = inputVars[:, 1:2, -1] [0]        #[1]  b~t
    assert (len(current_buffer_size) == 1)
    for i in current_buffer_size:
        used_inputs.add(i)
        print(i)

    past_chunk_throughput = inputVars[:, 2:3, :] [0][0]    #[k]  x~t
    print ("past_chunk_throughput",type(past_chunk_throughput))
    assert (len(past_chunk_throughput) == K)
    for i in past_chunk_throughput:
        used_inputs.add(i)
        print(i)

    past_chunk_download_time = inputVars[:, 3:4, :] [0][0]    #[k]  τ~t
    assert (len(past_chunk_download_time) == K)
    for i in past_chunk_download_time:
        used_inputs.add(i)
        print(i)


    next_chunk_sizes = inputVars[:, 4:5, :A_DIM] [0][0]       #[m]  n~t
    assert (len(next_chunk_sizes) == M)
    for i in next_chunk_sizes:
        used_inputs.add(i)
        print(i)

    number_of_chunks_left = inputVars[:, 4:5, -1] [0]       #[1]  c~t
    assert (len(number_of_chunks_left) == 1)
    for i in number_of_chunks_left:
        used_inputs.add(i)
        print(i)

    # split_0 = tflearn.fully_connected(inputs[:, 0:1, -1]    , 128, activation='relu')
    # split_1 = tflearn.fully_connected(inputs[:, 1:2, -1]    , 128, activation='relu')
    # split_2 = tflearn.conv_1d(inputs[:, 2:3, :]     , 128, 4, activation='relu')
    # split_3 = tflearn.conv_1d(inputs[:, 3:4, :]     , 128, 4, activation='relu')
    # split_4 = tflearn.conv_1d(inputs[:, 4:5, :A_DIM], 128, 4, activation='relu')
    # split_5 = tflearn.fully_connected(inputs[:, 4:5, -1]    , 128, activation='relu')

    # state = [np.zeros((S_INFO, S_LEN))]
    # state = np.roll(state, -1, axis=1)[0]
    # state[0, -1] = 1 # VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality                 # Last chunk bit rate -      [1]  l~t
    # state[1, -1] = 2 # buffer_size / BUFFER_NORM_FACTOR  # 10 sec #                                             # Current buffer size -      [1]  b~t
    # state[2, -1] = 3 # float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms                        # Past chunk throughput -    [k]  x~t
    # state[3, -1] = 4 #float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec                                      # Past chunk download time - [k]  τ~t
    # state[4, :A_DIM] = 5 # np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte                      # Next chunk sizes -         [m]  n~t
    # state[5, -1] = 6 #np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)  # Number of chunks left      [1]  c~t

    # last_chunk_bit_rate, current_buffer_size, past_chunk_throughput, past_chunk_download_time, next_chunk_sizes, number_of_chunks_left

    unsused_inputs = all_inputs - used_inputs
    print ("all inputs",all_inputs)
    print ("used inputs", used_inputs)
    print ("unused inputs", unsused_inputs)
    for var in unsused_inputs:
        l = 0
        u = 0
        network.setLowerBound(var, l)
        network.setUpperBound(var, u)

    # last_chunk_bit_rate
    for var in last_chunk_bit_rate:
        l = 1
        u = 2
        network.setLowerBound(var, l)
        network.setUpperBound(var, u)

    # current_buffer_size
    for var in current_buffer_size:
        l = 1
        u = 2
        network.setLowerBound(var, l)
        network.setUpperBound(var, u)

    # past_chunk_throughput
    for var in past_chunk_throughput:
        l = 1
        u = 2
        network.setLowerBound(var, l)
        network.setUpperBound(var, u)

    # past_chunk_download_time
    for var in past_chunk_download_time:
        l = 1
        u = 2
        network.setLowerBound(var, l)
        network.setUpperBound(var, u)

    # next_chunk_sizes
    for var in next_chunk_sizes:
        l = 1
        u = 1000
        network.setLowerBound(var, l)
        network.setUpperBound(var, u)

    # number_of_chunks_left
    for var in number_of_chunks_left:
        l = 1
        u = 2
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
