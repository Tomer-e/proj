import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.saved_model import tag_constants

MODEL_NAME = 'pcc_model'

# Freeze the graph

input_graph_path = MODEL_NAME+'.pb'
input_saved_model_dir = "."
checkpoint_path = './'+MODEL_NAME+'.ckpt'
input_saver_def_path = ""
input_binary = True
output_node_names = "O"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True
saved_model_tags = tag_constants.SERVING
input_graph_filename = None
output_graph_filename = "sdsdfgh"
input_meta_graph = False


freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                            input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_filename, clear_devices, "", "", "",
                              input_meta_graph, input_saved_model_dir,
                            saved_model_tags)




