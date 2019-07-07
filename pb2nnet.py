from NNet.converters.pb2nnet import pb2nnet
from tensorflow.python.saved_model import tag_constants

## Script showing how to run pb2nnet
# Min and max values used to bound the inputs
# inputMins  = [0.0,-3.141593,-3.141593,100.0,0.0]
# inputMaxes = [60760.0,3.141593,3.141593,1200.0,1200.0]

# Mean and range values for normalizing the inputs and outputs. All outputs are normalized with the same value
# means  = [1.9791091e+04,0.0,0.0,650.0,600.0,7.5188840201005975]
# ranges = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0,373.94992]

# Tensorflow pb file to convert to .nnet file
pbFile = '/cs/usr/tomerel/unsafe/VerifyingDeepRL/WP/model_relu_short/'#saved_model.pb'

# Convert the file
pb2nnet(pbFile, savedModel=True, savedModelTags=tag_constants.SERVING)
