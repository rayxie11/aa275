# Add path of lanenet directory
import sys
sys.path.append('C:\\Users\\ray_s\\Desktop\\Navigation for Autonomous Systems\\Project\\aa275_project\\lanenet\\tools')

import os
from test_lanenet import test_lanenet
from evaluate_lanenet_on_tusimple import eval_lanenet

current_dir = os.getcwd()
par_dir = os.path.dirname(current_dir)
img_path = par_dir+"/aa275_project/lanenet/1508987519235232.png"
#img_path = par_dir+"/data/img_data/"
w_path = par_dir+"/aa275_project/lanenet/pretrained_weights/tusimple_lanenet.ckpt"
test_lanenet(img_path,w_path)
#eval_lanenet(img_path,w_path,"results")

#img_files = os.listdir(par_dir+"/data/img_data")
#for img in img_files:
    #img_path = par_dir+"/data/img_data/"+img
    #test_lanenet(img_path,w_path,0)
#print(img_files)