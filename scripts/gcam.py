import numpy as np
import pandas as pd 
import tensorflow as tf
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc,roc_curve,accuracy_score
import matplotlib as mplbck
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage.transform import resize
mplbck.use('Agg')
mplbck.style.use(['fast'])

print(f'TF Version: {tf.__version__}')


""" this script will implement 3D gradcam and return the necessary heatmaps"""
parser = argparse.ArgumentParser()
parser.add_argument('-m','--model-path',dest='modelpath',type=str,required=True,help="The file path to read from - this should be the model in h5 format")
parser.add_argument('-t','--test-path',dest='test',type=str,required=True,help="The test data path to read from")
parser.add_argument('-sh',
	'--shape',dest='shape',type=str,required=True,help="What shape the input should be: decide from '1mm_subsamp' or '2mm_full'")
parser.add_argument('-l','--labels',dest='labels',type=str,required=True,help="The labels data path to read from")
parser.add_argument('-o','--outpath',dest='outpath',type=str,required=True,help="The path to write the heatmaps to")
parser.add_argument('-n','--name',dest='name',type=str,required=False,help="The layer to take activations from")
parser.add_argument('-auto','--auto',dest='lname',action='store_true',help="Whether to automatically take the last conv layer with a function in the script")
parser.add_argument('-norelu','--no-relu',dest='norelu',action='store_true',required=False, default=False,
help="If flagged, the resulting heatmap will not have been thresholded with a rectified linear unit (np.maximum(x,0))")

args=parser.parse_args()


# load data 
candidate = tf.keras.models.load_model(f'{args.modelpath}')
test_data = np.load(f'{args.test}')
with open(f'{args.labels}') as p:
    y_true = p.readlines()
y_true = [int(i.strip()) for i in y_true]
y_true = y_true[:101]
# scaling func 

sc = MinMaxScaler(feature_range=(0,255))
def scale_data_255(patient):

        return sc.fit_transform(patient)
# zeros vec 

scale_data = np.zeros(test_data.shape)

# determine shape scheme 
if args.shape == '1mm_subsamp':
	x = 120
	y = 140
if args.shape == '2mm_full':
	x = 91
	y = 109

test_data = test_data.reshape(101,x,y,x,1)

scale_data = np.zeros(test_data.shape)

for i in range(test_data.shape[0]):
    scale_data[i] = scale_data_255(test_data[i].flatten().reshape(-1,1)).reshape(x,y,x,1)

# data scaled and ready for model - define gradcam func 

def grad_cam_3d_g(layer_name,model,data,norelu=False):
    
    # create gradient model with a) model inputs, b) desired conv layer output, c) model outputs
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    # evaluate gradients with automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([data]))
        loss = predictions[:,0]
    
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    
    # now create cam 
    # this just changes type (and applies mask) for the guided output
    # get mean along feature maps 
    weights = np.mean(grads, axis=(0, 1, 2))
    # create cam
    cam = np.zeros(output.shape[0:3], dtype = np.float32)
    # the cam is then weighted by the gradients * outputs  
    for i, w in enumerate(weights):
        cam += w * output[:, :, :, i]
    # bilinear interpolation to input size after scaling between 0 and 1 
    if norelu==False:
        cam = np.maximum(cam.numpy(),0)
    cam = cam / np.max(cam)
    cam = resize(cam,(x,y,x)) 
    
    heatmap = (cam - cam.min()) / (cam.max() - cam.min()) * 255.0
    
    
    
    return heatmap
assert args.name or args.lname==True, "Either provide a layer name or select the --auto flag!"

norelu = args.norelu
def get_layer_name(model):
    return [layer.name for layer in model.layers if 'conv' in layer.name][-1]
if args.lname:
    layer_name = get_layer_name(candidate)
else:
    layer_name = args.name
heatmaps = [
    grad_cam_3d_g(layer_name=layer_name,model=candidate,data=scale_data[i],norelu=norelu) for i in range(101)
]

heatmaps = np.array(heatmaps)
np.save(f'{args.outpath}',heatmaps) 
