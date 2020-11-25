# [filter size, stride, padding]
# Assume the two dimensions are the same
# Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
# 
# Each layer i requires the following parameters to be fully represented: 
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i

import math
convnet = [[5,2,1],[5,2,1],[5,2,1],[5,1,1],[5,1,1]]
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
imsize = 256

def outFromIn(conv, layerIn):
	n_in = layerIn[0]
	j_in = layerIn[1]
	r_in = layerIn[2]

	k = conv[0]
	s = conv[1]
	p = conv[2]

	n_out = math.floor((n_in - k + 2*p)/s) + 1
	actualP = (n_out-1)*s - n_in + k 
	pR = math.ceil(actualP/2)
	pL = math.floor(actualP/2)

	j_out = j_in * s
	r_out = r_in + (k - 1)*j_in
	return n_out, j_out, r_out
	
def printLayer(layer, layer_name):
	print(layer_name + ":" + "\t receptive size: %s \t " % (layer[2]))
	print()
 
layerInfos = []

if __name__ == '__main__':
# first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1;
	print ("-------Net summary------")
	currentLayer = [imsize, 1, 1]
	# printLayer(currentLayer, "input image")
	for i in range(len(convnet)):
		currentLayer = outFromIn(convnet[i], currentLayer)
		layerInfos.append(currentLayer)
		printLayer(currentLayer, layer_names[i])
	print ("------------------------")
	