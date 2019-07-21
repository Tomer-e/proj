import tensorflow as tf
import saved_model_cli as util
from NNet.utils import writeNNet as writer
# import conversion.writeNNet as writer
import numpy as np


def loader(model_dir):
    if not tf.saved_model.maybe_saved_model_directory(model_dir):
        exit()

    with tf.Session(graph=tf.Graph()) as sess:
        a = tf.saved_model.loader.load(sess, tags=['serve'], export_dir=model_dir)

        # print(a)
        network_params = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/pi')
        # print(network_params)
        layers = sess.run(network_params)
        # print(sess.graph_def.nodes)
        layer_counter = 0
        temp_weight = -1
        weights = []
        biases = []


        # gd = sess.graph.as_graph_def()
        #
        # for node in gd.node:
        #     print(node.name)


        for i in range(len(layers)):
            if i % 2 == 0:
                temp_weight_shape = layers[i].shape
                weights.append(layers[i])
            if i % 2 != 0:
                layers_bias_shape = layers[i].shape
                biases.append(layers[i])
                print("Layer {}:\n    weights:{}\n    bias:{}\n ".format(
                    layer_counter, temp_weight_shape, layers_bias_shape))
                layer_counter += 1

        means = np.zeros((31))  # inputs+1 output
        means.fill(0)
        means[30] = 0
        ranges = np.ones((31))  # inputs+1 output
        ranges[30] = 255
        # writer.writeNNet(weights[:-1], biases, np.zeros((30)), np.ones((30)), means, ranges, model_dir+"/out/model_2.nnet")

        meta_graph_def = util.get_meta_graph_def(model_dir, 'serve')

        inputs_tensor_info = util._get_inputs_tensor_info_from_meta_graph_def(
            meta_graph_def, 'serving_default')

        print(inputs_tensor_info)
        input_arr = np.zeros((1,30))
        inputs_feed_dict = {"input/Ob:0": input_arr}
      # Get outputs
        outputs_tensor_info = util._get_outputs_tensor_info_from_meta_graph_def(
            meta_graph_def,  'serving_default')
      # Sort to preserve order because we need to go from value to key later.
        output_tensor_keys_sorted = sorted(outputs_tensor_info.keys())
        output_tensor_names_sorted = [
          outputs_tensor_info[tensor_key].name
          for tensor_key in output_tensor_keys_sorted
    ]
        outputs = sess.run(output_tensor_names_sorted, feed_dict=inputs_feed_dict)
        # np.save(model_dir + "/sample_input.npy", input_arr)

        print(outputs)
        print(outputs[0]*0.25)
        print(outputs[1]*0.25)
        print(np.tanh(outputs))


        inputs_feed_dict["input/Ob:0"].fill(0.5)
        outputs = sess.run(output_tensor_names_sorted,
                           feed_dict=inputs_feed_dict)
        # np.save(model_dir + "/sample_input.npy", input_arr)

        print(outputs)
        print(outputs[0]*0.25)
        print(outputs[1]*0.25)
        print(np.tanh(outputs))

        inputs_feed_dict["input/Ob:0"].fill(1)
        outputs = sess.run(output_tensor_names_sorted,
                           feed_dict=inputs_feed_dict)
        # np.save(model_dir + "/sample_input.npy", input_arr)

        print(outputs)
        print(outputs[0]*0.25)
        print(outputs[1]*0.25)
        print(np.tanh(outputs))

import sys

def main():
    if len(sys.argv)!=2:
        print("usage:",sys.argv[0], "<model_dir>")
        exit(0)

    model_dir= sys.argv[1]
    loader(model_dir)


if __name__ == "__main__":
    main()
