#%% Save 
import mrcfile
import numpy as np
import tifffile as tiff
from skimage.io import imsave
import matplotlib.pyplot as plt
dataPath = "dataset"
def find_mrc_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".mrc"):
                yield os.path.join(root, file)
                
delete_mrc=1
for mrc_file in find_mrc_files(dataPath):
    print(mrc_file)
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        mrc_data = mrc.data
        mrc_data = np.transpose(mrc_data)
    imsave(mrc_file[:mrc_file.index(".")]+".tif", mrc_data, imagej=True)
    if delete_mrc:
        os.remove(mrc_file)
        

#%% Move dataset
import shutil
dataset_to_move="dataset/CCPs" #all the data are stored slightly differently. Here,we pick CCPs.
data_path_fin="dataset/ffhq_502_1004"

hrPath=os.path.join(data_path_fin,"hr_1004")
lrPath=os.path.join(data_path_fin,"lr_502")
srPath=os.path.join(data_path_fin,"sr_502_1004")
try:
    os.mkdir(data_path_fin)
    os.mkdir(hrPath)
    os.mkdir(srPath)
    os.mkdir(lrPath)
except:
    pass


level = "01" #level of down-sampling we want to consider
folders=os.listdir(dataset_to_move)
folders=[f for f in folders if "." not in f]
for folder in folders:
    files = os.listdir(os.path.join(dataset_to_move,folder))
    for file in files:
        if file=="SIM_gt.tif": #can change to move if you do not want duplicate data
            shutil.copy(os.path.join(dataset_to_move,folder,file),os.path.join(hrPath,folder+"SIM.tif"))
            shutil.copy(os.path.join(dataset_to_move,folder,file),os.path.join(srPath,folder+"SIM.tif"))
        elif level in file:
            shutil.copy(os.path.join(dataset_to_move,folder,file),os.path.join(lrPath,folder+"SIM.tif"))
#%% Make crops
import os
import tifffile as tiff

def cropImage(image,crop_size):
    cut = int((image.shape[0] % crop_size)/2)
    image=image[cut:-cut,cut:-cut,:]
    croppedImages=[]
    for i in range(0,image.shape[0],crop_size):
        for j in range(0,image.shape[1],crop_size):
            croppedImages.append(image[i:i+crop_size,j:j+crop_size,:])
    return croppedImages

crop_size=64
dataPath="dataset/ffhq_502_1004"
hrPath=os.path.join(dataPath,"hr_1004")
lrPath=os.path.join(dataPath,"lr_502")
srPath=os.path.join(dataPath,"sr_502_1004")

newPath="dataset/ffhq_"+str(crop_size)+"_"+str(crop_size*2)
newHrPath=os.path.join(newPath,"hr_128")
newSrPath=os.path.join(newPath,"sr_64_128")
newLrPath=os.path.join(newPath,"lr_64")

try:
    os.mkdir(newHrPath)
    os.mkdir(newSrPath)
    os.mkdir(newLrPath)
    os.mkdir(newPath)
except:
    print("newPath already exists")

for img in os.listdir(hrPath):
    filename = img[:img.index(".tif")]
    
    hr_img=tiff.imread(os.path.join(hrPath,img))
    lr_img=tiff.imread(os.path.join(lrPath,img))
    sr_img=tiff.imread(os.path.join(srPath,img))
    
    croppedImages_hr=cropImage(np.expand_dims(hr_img,-1),crop_size*2)
    croppedImages_sr=cropImage(np.expand_dims(sr_img,-1),crop_size*2)
    croppedImages_lr=cropImage(lr_img,crop_size)
    
    for i in range(0,len(croppedImages_hr)):
        tiff.imsave(os.path.join(newHrPath,filename+str(i)+".tif"),croppedImages_hr[i])
        tiff.imsave(os.path.join(newSrPath,filename+str(i)+".tif"),croppedImages_sr[i])
        tiff.imsave(os.path.join(newLrPath,filename+str(i)+".tif"),croppedImages_lr[i])


    
    
    
#%% Read and plot multi-channel tiff files
import tifffile as tiff
import matplotlib.pyplot as plt
dataPath = "dataset/CCPs/train/Cell_001/RawSIMData_level_01.tif"

test = tiff.imread(dataPath)
print(test.shape)
print(test.dtype)
print(test[0,0,0])
plt.imshow(test[:,:,0])
plt.show()