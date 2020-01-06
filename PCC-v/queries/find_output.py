
import numpy as np
from eval_network import evaluateNetwork
from tensorflow.python.saved_model import tag_constants


outputs = ["model/flatten/Shape",
"model/flatten/strided_slice/stack",
"model/flatten/strided_slice/stack_1",
"model/flatten/strided_slice/stack_2",
"model/flatten/Reshape/shape",
"model/flatten/Reshape",
"model/pi_fc0/w/Initializer/initial_value",
"model/pi_fc0/w",
"model/pi_fc0/w/Assign",
"model/pi_fc0/w/read",
"model/pi_fc0/b/Initializer/Const",
"model/pi_fc0/b",
"model/pi_fc0/b/Assign",
"model/pi_fc0/b/read",
"model/pi_fc0/MatMul",
"model/pi_fc0/add",
"model/Relu",
"model/vf_fc0/w/Initializer/initial_value",
"model/vf_fc0/w",
"model/vf_fc0/w/Assign",
"model/vf_fc0/w/read",
"model/vf_fc0/b/Initializer/Const",
"model/vf_fc0/b",
"model/vf_fc0/b/Assign",
"model/vf_fc0/b/read",
"model/vf_fc0/MatMul",
"model/vf_fc0/add",
"model/Relu_1",
"model/pi_fc1/w/Initializer/initial_value",
"model/pi_fc1/w",
"model/pi_fc1/w/Assign",
"model/pi_fc1/w/read",
"model/pi_fc1/b/Initializer/Const",
"model/pi_fc1/b",
"model/pi_fc1/b/Assign",
"model/pi_fc1/b/read",
"model/pi_fc1/MatMul",
"model/pi_fc1/add",
"model/Relu_2",
"model/vf_fc1/w/Initializer/initial_value",
"model/vf_fc1/w",
"model/vf_fc1/w/Assign",
"model/vf_fc1/w/read",
"model/vf_fc1/b/Initializer/Const",
"model/vf_fc1/b",
"model/vf_fc1/b/Assign",
"model/vf_fc1/b/read",
"model/vf_fc1/MatMul",
"model/vf_fc1/add",
"model/Relu_3",
"model/vf/w/Initializer/initial_value",
"model/vf/w",
"model/vf/w/Assign",
"model/vf/w/read",
"model/vf/b/Initializer/Const",
"model/vf/b",
"model/vf/b/Assign",
"model/vf/b/read",
"model/vf/MatMul",
"model/vf/add",
"model/pi/w/Initializer/initial_value",
"model/pi/w",
"model/pi/w/Assign",
"model/pi/w/read",
"model/pi/b/Initializer/Const",
"model/pi/b",
"model/pi/b/Assign",
"model/pi/b/read",
"model/pi/MatMul",
"model/pi/add",
"model/pi/logstd/Initializer/zeros",
"model/pi/logstd",
"model/pi/logstd/Assign",
"model/mul",
"model/add",
"model/concat/axis",
"model/concat",
"model/q/w/Initializer/initial_value",
"model/q/w",
"model/q/w/Assign",
"model/q/w/read",
"model/q/b/Initializer/Const",
"model/q/b",
"model/q/b/Assign",
"model/q/b/read",
"model/q/MatMul",
"model/q/add",
"model/Const",
"model/split/split_dim",
"model/split",
"model/Exp",
"output/Shape",
"output/random_normal/mean",
"output/random_normal/stddev",
"output/random_normal/RandomStandardNormal",
"output/random_normal/mul",
"output/random_normal",
"output/mul",
"output/add",
"output/sub",
"output/truediv",
"output/Square",
"output/Sum/reduction_indices",
"output/Sum",
"output/mul_1/x",
"output/mul_1",
"output/Shape_1",
"output/strided_slice/stack",
"output/strided_slice/stack_1",
"output/strided_slice/stack_2",
"output/strided_slice",
"output/Cast",
"output/mul_2/x",
"output/mul_2",
"output/add_1",
"output/Sum_1/reduction_indices",
"output/Sum_1",
"output/add_2",
"output/strided_slice_1/stack",
"output/strided_slice_1/stack_1",
"output/strided_slice_1/stack_2",
"output/strided_slice_1"]
def create_network(filename):
    output_op_name = "output/Shape"
    input_op_names = ["input/Ob"]
    # read_tf(filename, inputName=None, outputName=None, savedModel=False, savedModelTags=[]):


    #########################################################################################
    sanity_inputs =[0.,        1.,         1.  ,       0.,         1.,         1.04      ,
          0.,        1.,         1.04,       0.,         1.,         1.22727273,
          0.,        1.,         1.04,       0.,         1.,         1.28571429,
          0.,        1.,         1.08,       0.,         1.,         1.        ,
          0.,        1.,         1.04,       0.,         1.,         1.08]
    # Applying new delta 0.418537
    # old delta = 1.652943
    #########################################################################################

    latency_inflation = []
    Latency_Ratio = []
    Send_Ratio = []
    for i in range (len (sanity_inputs)):
        if i%3 == 0:
            latency_inflation.append(sanity_inputs[i])
        elif i%3 == 1:
            Latency_Ratio.append(sanity_inputs[i])
        else:
            Send_Ratio.append(sanity_inputs[i])

    print(latency_inflation)
    print(Latency_Ratio)
    print(Send_Ratio)

    sanity_inputs = latency_inflation + Latency_Ratio + Send_Ratio


    sanity_inputs = np.asanyarray(sanity_inputs).reshape ((1,30))
    print("my inputs:", sanity_inputs)
    # exit (0)

    for output_op_name in outputs:
        try:
            output = evaluateNetwork(filename, sanity_inputs, input_op_names, output_op_name)
            print ("==========",output_op_name,"==========")
            print("network output:",output )
        except IndexError:
            print("error with",output_op_name )
    exit (0)
    return network, input_op_names, output_op_name

# Network inputs:
# 0 - 9   : latency gradient, the derivative of latency with respect to time
# 10 - 19 : latency ratio, the ratio of the current MI’s mean latency to minimum observed mean latency of any MI in
#           the connection’s history
# 20 - 29 : sending ratio, the ratio of packets sent to packets acknowledged by the receiver

def basic_test(filename, to_log_file):

    network,input_op_names, output_op_name =  create_network(filename)

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
