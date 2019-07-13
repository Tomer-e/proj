from NNet.converters.pb2nnet import pb2nnet
from tensorflow.python.saved_model import tag_constants
import sys
import numpy as np


## Script showing how to run pb2nnet
# Min and max values used to bound the inputs
# inputMins  = [0.0,-3.141593,-3.141593,100.0,0.0]
# inputMaxes = [60760.0,3.141593,3.141593,1200.0,1200.0]

# Mean and range values for normalizing the inputs and outputs. All outputs are normalized with the same value
# means  = [1.9791091e+04,0.0,0.0,650.0,600.0,7.5188840201005975]
# ranges = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0,373.94992]

# Tensorflow pb file to convert to .nnet file
# pbFile = '/cs/usr/tomerel/unsafe/VerifyingDeepRL/WP/model_relu_short/frozen_graph_2.pb'  #saved_model.pb'

# Convert the file
# pb2nnet(pbFile, savedModel=True, savedModelTags=tag_constants.SERVING)


# pb2nnet(pbFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName="", savedModel=False, savedModelTags=[]):
def main():
    if len(sys.argv)!= 2:
        print("usage:",sys.argv[0], "<pbFile_name>")
        exit(0)
    pbFile = sys.argv[1]

    inputMins = np.zeros((30))
    inputMaxes = np.ones((30))

    # Mean and range values for normalizing the inputs and outputs. All outputs are normalized with the same value
    means = np.zeros((31))  # inputs+1 output
    means.fill(0.5)
    ranges = np.ones((31))  # inputs+1 output
    ranges[30] = 255

    inputName = 'input/Ob'
    outputName = 'model/split'
    pb2nnet(pbFile,inputMins=inputMins, inputMaxes=inputMaxes, means=means, ranges=ranges, inputName=inputName,outputName=outputName)


if __name__ == "__main__":
    main()


