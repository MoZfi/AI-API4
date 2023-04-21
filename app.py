# the print function inside each route

from flask import Flask, jsonify, request,send_file,make_response
from flask_restful import Api, Resource
import csv
from PIL import Image
import io
import json
import base64


'''AI library'''
# first you shuld install : gco-wrapper
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms #transforms.ToTensor() changes image to tensor
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import pickle
import mixup
from mixup import mixup_graph
import threading # for solving the main thread


os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%matplotlib inline

app = Flask(__name__)
api = Api(app) 

       
@app.route('/')
def hello_world():
    return "Hello World!"

# Puzzlemix
@app.route('/Puzzlemix', methods=['POST'], endpoint='puzzle')
def puzzle():
    if request.method == 'POST':
        files = request.files
        imgs = files.getlist('image')
        file_name = [img.filename for img in imgs]
        imgs = [i.read() for i in imgs] # this will read binary images

        rs = {} # it is a dict?
        for i in range(len(file_name)):
            rs[i] = file_name[i]

        print (file_name)
        name1 = rs[0]
        name2 = rs[1]
        print (rs) # {0: 'ILSVRC2012_val_00000324.JPEG', 1: 'ILSVRC2012_val_00000318.JPEG'}
        #return jsonify(rs)

        with open('img_name_label_testset.csv') as file_obj:

            reader_obj = csv.reader(file_obj)
            lable1 = None
            lable2 = None
            for row in reader_obj:
                if row[1] == name1 and lable1 is None:
                    lable1 = row[3]
                if row[1] == name2 and lable2 is None:
                    lable2 = row[3]
                if lable1 is not None and lable2 is not None:
                    break

        rn= {}
        rn[0] = lable1
        rn[1] = lable2
        print (rn) # {0: '21', 1: '7'}
        #return jsonify(rn)


        for file_name in files:
            file = files[file_name]
            print(f"File Name: {file.filename}")
            print(f"File Content: {file.read()}")
        #return "Files received"
    #else:
        #return "Please send a POST request"



        # Get the uploaded images from the request
        img1 = imgs[0]
        img2 = imgs[1]
        #print(img1)
        #return jsonify(rn) # just for check the print in the terminal

        # Open the images using Pillow
        # before it need :pip install Pillow
        image1 = Image.open(io.BytesIO(img1))
        image2 = Image.open(io.BytesIO(img2))
        #return jsonify(rn) #---> check point

        # chek the images --> convert to array
        img1_array = np.array(image1)
        img2_array = np.array(image2)
        print(img1_array.shape) #--> (500, 375, 3)
        #return jsonify(rn)
        #return img1_array.tobytes(), img2_array.tobytes()



        ############### Model ###############
        resnet = models.resnet18(pretrained=True)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        mean_torch = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std_torch = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        resnet.eval()
        criterion = nn.CrossEntropyLoss()

        ### Data:  imagenet with transform ####
        img_exists = True

        if img_exists:
        # I used this codes to load data
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        else:
            pass

        sample_num = 2

        tensor1=test_transform(image1)
        tensor2=test_transform(image2)
        print(tensor1.shape)
        ### Selected Examples
        input_sp = torch.stack([tensor1, tensor2], dim=0)
        targets = torch.tensor([int(lable1), int(lable2)])


        #print_fig((input_sp * std_torch + mean_torch)[:sample_num])


        ########### Saliency ###############
        input_var = input_sp[:sample_num].clone().detach().requires_grad_(True)
        output = resnet(input_var)
        loss = criterion(output, targets[:sample_num])
        loss.backward()

        unary = torch.sqrt(torch.mean(input_var.grad **2, dim=1))
        unary = unary / unary.view(sample_num, -1).max(1)[0].view(sample_num, 1, 1)
        #print_fig(unary)

        unary16 = F.avg_pool2d(unary, 224//16)
        unary16 = unary16 / unary16.view(sample_num, -1).max(1)[0].view(sample_num, 1, 1)
        #print_fig(unary16)

        """### Puzzle Mix"""

        #### Alpha Sweep###
        indices = [1,0]

        n_labels = 3
        block_num = 16

        alpha = 0.5
        beta = 0.8
        gamma = 1.0
        eta = 0.2

        transport = False

        for alpha in np.linspace(0,1,5):
            output = mixup_graph(input_sp, unary, indices=indices, n_labels=n_labels,
                                block_num=block_num, alpha=np.array([alpha]).astype('float32'), beta=beta, gamma=gamma, eta=eta,
                                neigh_size=2, mean=mean_torch, std=std_torch,
                                transport=transport, t_eps=0.8, t_size=16,
                                device='cpu')

           

        ###Beta Sweep###
        indices = [1,0]

        n_labels = 3
        block_num = 16

        alpha = 0.5
        gamma = 0.
        eta = 0.2

        transport = False

        for beta in np.linspace(0,0.8,4):
            output = mixup_graph(input_sp, unary, indices=indices, n_labels=n_labels,
                                block_num=block_num, alpha=np.array([alpha]).astype('float32'), beta=beta, gamma=gamma, eta=eta,
                                neigh_size=2, mean=mean_torch, std=std_torch,
                                transport=transport, t_eps=0.8, t_size=16,
                                device='cpu')

           

        ### Gamma Sweep####
        indices = [1,0]

        n_labels = 3
        block_num = 16

        alpha = 0.5
        beta = 0.2
        eta = 0.2

        transport = False

        for gamma in np.linspace(0,2,4):
            output = mixup_graph(input_sp, unary, indices=indices, n_labels=n_labels,
                                block_num=block_num, alpha=np.array([alpha]).astype('float32'), beta=beta, gamma=gamma, eta=eta,
                                neigh_size=2, mean=mean_torch, std=std_torch,
                                transport=transport, t_eps=0.8, t_size=16,
                                device='cpu')

            

        ###Transport ###
        indices = [1,0]

        n_labels = 2
        block_num = 4

        alpha = 0.4
        beta = 0.2
        gamma = 1.0
        eta = 0.2

        transport = True
        t_eps=0.2
        t_size=224//block_num

        output = mixup_graph(input_sp, unary, indices=indices, n_labels=n_labels,
                            block_num=block_num, alpha=np.array([alpha]).astype('float32'), beta=beta, gamma=gamma, eta=eta,
                            neigh_size=2, mean=mean_torch, std=std_torch,
                            transport=transport, t_eps=t_eps, t_size=t_size,
                            device='cpu')

       
        
        print(output[0].shape)
        
        # Convert the outout tensor to a NumPy array
        input_me = output [0]* std_torch + mean_torch

        ########## 1- create outout as an array
        np_array = input_me.numpy()
        ratio= output[1][0]
        print(ratio)
        shuffled_targets= ratio * targets[0] + targets[1]* (1 - ratio)
        shuffled_targets=shuffled_targets.numpy()
        #print(shuffled_targets)
        #plt.imshow(np_array.transpose(1, 2, 0))
        #plt.show()

         
        # Convert the NumPy array to a Python dictionary
        tensor_dict = {'message': 'Image processed successfully!', 'lable':shuffled_targets.tolist() ,'data': np_array.tolist()}
        # Convert the Python dictionary to JSON
        tensor_json = json.dumps(tensor_dict)
        
        

        # Return the JSON response
        #return jsonify(tensor_json)
        
        ########### 2- plot the outout image
        # create GUI for the output image using Matplotlib
        fig, axes = plt.subplots(1,len(input_me),figsize=(3*len(input_me),3))

        for i, ax in enumerate(axes):
            if len(input_me.shape) == 4:
                ax.imshow(input_me[i].permute(1,2,0).numpy())
            else :
                ax.imshow(input_me[i].numpy(), cmap='gray', vmin=0., vmax=1.)

        
        plt.subplots_adjust(wspace = 0.1)

        #plt.savefig('image.png')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        buf.seek(0)

        response = make_response(buf.getvalue())
        response.headers.set('Content-Type', 'image/png')
        response.headers.set('Content-Disposition', 'attachment', filename='image_Puzzlemix.png')
        return response
        
        
        #'''

###############Cutmix
@app.route('/Cutmix', methods=['POST'],  endpoint='cut')
def cut():
    if request.method == 'POST':
        files = request.files
        imgs = files.getlist('image')
        file_name = [img.filename for img in imgs]
        imgs = [i.read() for i in imgs] # this will read binary images

        rs = {} 
        for i in range(len(file_name)):
            rs[i] = file_name[i]

        name1 = rs[0]
        name2 = rs[1]
        print (rs) # {0: 'ILSVRC2012_val_00000324.JPEG', 1: 'ILSVRC2012_val_00000318.JPEG'}
        #return jsonify(rs)

        with open('img_name_label_testset.csv') as file_obj:

            reader_obj = csv.reader(file_obj)
            lable1 = None
            lable2 = None
            for row in reader_obj:
                if row[1] == name1 and lable1 is None:
                    lable1 = row[3]
                if row[1] == name2 and lable2 is None:
                    lable2 = row[3]
                if lable1 is not None and lable2 is not None:
                    break

        rn= {}
        rn[0] = lable1
        rn[1] = lable2
        print (rn) # {0: '21', 1: '7'}
        #return jsonify(rn)


        for file_name in files:
            file = files[file_name]
            print(f"File Name: {file.filename}")
            print(f"File Content: {file.read()}")
        #return "Files received"
    #else:
        #return "Please send a POST request"



        # Get the uploaded images from the request
        img1 = imgs[0]
        img2 = imgs[1]
        #return jsonify(rn) # just for check the print in the terminal

        # Open the images using Pillow
        # before it need :pip install Pillow
        image1 = Image.open(io.BytesIO(img1))
        image2 = Image.open(io.BytesIO(img2))
        #return jsonify(rn) #---> check point

        # chek the images --> convert to array
        img1_array = np.array(image1)
        img2_array = np.array(image2)
        print(img1_array.shape) #--> (500, 375, 3)
        #return jsonify(rn)
        #return img1_array.tobytes(), img2_array.tobytes()


        ### Data: imagenet with transform ####
        img_exists = True

        if img_exists:
        # I used this codes to load data
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(254),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        else:
            pass

        sample_num = 2


        tensor1=test_transform(image1)
        tensor2=test_transform(image2)
        print(tensor1.shape)
        ### Selected Examples
        input_sp = torch.stack([tensor1, tensor2], dim=0)
        targets = torch.tensor([int(lable1), int(lable2)])



        """### Cut Mix"""

        alpha=0.5
        data=input_sp
        _, _, height, width = data.shape
        print(data.size(0))
        indices=[1,0]
        shuffled_data = data[indices]
        shuffled_targets =[]
        sorted_indexes = torch.argsort(targets, descending=True)
        shuffled_targets = torch.index_select(targets, index=sorted_indexes,dim=0,)

        lam = np.random.beta(alpha, alpha)

        image_h, image_w = data.shape[2:] #width and height of image
        cx = np.random.uniform(0, image_w) #  rx, ry image--> center bounding box
        cy = np.random.uniform(0, image_h) #  rx, ry image--> center bounding box
        w = image_w * np.sqrt(1 - lam) #  rw, rh yani size bounding box 
        h = image_h * np.sqrt(1 - lam) #  rw, rh yani size bounding box 
        x0 = int(np.round(max(cx - w / 2, 0))) # coordination  B
        x1 = int(np.round(min(cx + w / 2, image_w))) # coordination  B'
        y0 = int(np.round(max(cy - h / 2, 0))) # coordination  B
        y1 = int(np.round(min(cy + h / 2, image_h))) # coordination  B'

        data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
        lam = 1 - ((x1 - x0) * (y1 - y0) / (width * height))

        shuffled_targets1= lam * targets[0] + targets[1]* (1 - lam)
        targets1 = (targets,shuffled_targets,shuffled_targets1, lam)

        print(targets1)
        print(data[0].shape)
        
        ########## 1- create outout as an array
        # Convert the tensor to a NumPy array
        input_me = data [:2]
        np_array = input_me.numpy()
        shuffled_targets1_1=shuffled_targets1.numpy()

        # Convert the NumPy array to a Python dictionary
        tensor_dict = {'message': 'Image processed successfully!','lable':shuffled_targets1_1.tolist(),'data': np_array.tolist()}

        # Convert the Python dictionary to JSON
        tensor_json = json.dumps(tensor_dict)

        # Return the JSON response
        #return jsonify(tensor_json)


        ########### 2- plot the outout image
        
        # create GUI for the output image using Matplotlib
        image_final= (data * std_torch + mean_torch)[:sample_num]# tensor of size [2, 3, 224, 224]
      
        fig, axes = plt.subplots(1,len(image_final),figsize=(3*len(image_final),3))

        # axes is a single Axes object
        for i, ax in enumerate(axes):
            if len(image_final.shape) == 4:
                ax.imshow(image_final[i].permute(1,2,0).numpy())
            else :
                ax.imshow(image_final[i].numpy(), cmap='gray', vmin=0., vmax=1.)
        
        plt.subplots_adjust(wspace = 0.1)

        #plt.savefig('image.png')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')

        buf.seek(0)

        response = make_response(buf.getvalue())
        response.headers.set('Content-Type', 'image/png')
        response.headers.set('Content-Disposition', 'attachment', filename='image_cutmix.png')
        return response
        
         
        #'''
   

if __name__=="__main__":
    app.run(host='0.0.0.0',port=int("5000"), debug=True)
    

    
     
    
