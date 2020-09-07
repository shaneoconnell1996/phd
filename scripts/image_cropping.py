# this script will implement image cropping via interpolation.  

import numpy as np
import nibabel as nib
import glob
import argparse
import itertools
# load in the data
def read_nifti_image(path): # read images from path
    temp=nib.load(path)
    image=temp.get_fdata() # parse to image
    return(image)
parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',dest='path',type=str,required=True,help="The file path to read from")
parser.add_argument('-o','--outpath',dest='outpath',type=str,required=True,help="The path to write the file to")
parser.add_argument('-x','--xshape',dest='x',type=int,required=True,help="Shape of X arg")
parser.add_argument('-y','--yshape',dest='y',type=int,required=True,help="Shape of Y arg")
parser.add_argument('-z','--zshape',dest='z',type=int,required=True,help="Shape of Z arg")

args=parser.parse_args()

filelist = glob.glob(f'{args.path}*nii.gz')

# what I think is happening in the function below (thanks researchgate) is by specifying the new size you want, you can end up getting back your cropped image 
# through taking the delta (orig_x / new_x). This delta is then used to take the cropped representation of the pixels through indexing the original image

total_df_numpy=np.array([read_nifti_image(fname) for fname in filelist])
def resize_data_stack(data,x,y,z):
    initial_size_x = data.shape[0]
    initial_size_y = data.shape[1]
    initial_size_z = data.shape[2]

    new_size_x = x
    new_size_y = y
    new_size_z = z

    delta_x = initial_size_x / new_size_x
    delta_y = initial_size_y / new_size_y
    delta_z = initial_size_z / new_size_z

    new_data = np.zeros((new_size_x, new_size_y, new_size_z))

    for x, y, z in itertools.product(range(new_size_x),
                                     range(new_size_y),
                                     range(new_size_z)):
        new_data[x][y][z] = data[int(x * delta_x)][int(y * delta_y)][int(z * delta_z)]

    return new_data
shape = (total_df_numpy.shape[0],args.x,args.y,args.z)
new_dataset=np.zeros(shape)
for i in range(total_df_numpy.shape[0]):
    new_dataset[i] = resize_data_stack(total_df_numpy[i],args.x,args.y,args.z)

# write 

np.save(f'{args.outpath}.npy',new_dataset)
