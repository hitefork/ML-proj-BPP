import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from operator import *
import copy
import random

def get_rotation(data,index):
    rotation=[]
    if index == 0:
        rotation= [0, 1, 2]
    elif index == 1:
        rotation= [0, 2, 1]
    elif index == 2:
        rotation= [1, 0, 2]
    elif index == 3:
        rotation= [1, 2, 0]
    elif index == 4:
        rotation= [2, 0, 1]
    elif index == 5:
        rotation= [2, 1, 0]
    temp=np.copy(data)
    temp=temp[rotation]
    return temp

def get_inverse_rotation(data,index):
    mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 4,
        4: 3,
        5: 5
    }
    return get_rotation(data,mapping[index])

class BoxMaker():
    '''
    To make expert data set for solving 3D tetris
    '''
    def __init__(self,ldc_ht=100,ldc_wid=100,ldc_len=100,seed: int =42,print=0):
        self.ldc_ht  = ldc_ht
        self.ldc_wid = ldc_wid
        self.ldc_len = ldc_len
        self.seed = seed
        self.print = print



    def get_boxes(self,randomseed=True,israndomsort=False,isprint=False):
        """Generates instances of the 2D and 3D bin packing problems

        Parameters
        ----------
        bin_size: List[int], optional (default = [10,10,10])
            List of length 2 or 3 with the dimensions of the container (default = (10,10,10))

        Returns
        -------
        List[np.array(np.int32)]
        A list of length num_items with the dimensions of the randomly generated boxes.
        
        List[dict]
        A list of items.
        """
        if randomseed==False:
            np.random.seed(self.seed)
        
        item_original = {
            'shape':    np.array([self.ldc_len,self.ldc_wid,self.ldc_ht]),  #[length, height, width]
            'position': np.array([0, 0, 0]),   #[x, y, z]
            'rotation':  0,    
        }    
        """
        rotation: 
        [0, 1, 2]->0
        [0, 2, 1]->1
        [1, 0, 2]->2
        [1, 2, 0]->3
        [2, 0, 1]->4
        [2, 1, 0]->5   
        """            
        # initialize item list
        items = []
        
        items.append(item_original)
        
        N=np.random.randint(10,51)

        while len(items) < N:
            # pop an item from the list
            item=items.pop(0)
            # choose an axis of the item randomly
            axis=np.random.randint(0,3)

            if item['shape'][axis] <=1:
                items.append(item)
                continue
            # Choose a position randomly on the axis by the distance to the center of edge
            if item['shape'][axis]<=3:
                position=1
            else:
                position=np.random.randint(1,item['shape'][axis]/2)

            # Split the item into two items.
            items1=copy.deepcopy(item)
            items2=copy.deepcopy(item)

    
            items1['shape'][axis]=position
            items2['shape'][axis]=items2['shape'][axis]-position


            items2['position'][axis]=items2['position'][axis]+position

            items.append(items1)
            items.append(items2)
        
        
        
        for i in range(len(items)):
            shuffle_i = np.random.randint(0,6)
            items[i]['rotation']= shuffle_i
            items[i]['shape'] = get_rotation(items[i]['shape'],shuffle_i)


        
        
        if israndomsort:
            floor_building_breadth = 0
            floor_building_length  = 1
            wall_building_length   = 2
            wall_building_breadth  = 3

            building_choice = np.random.randint(0,4,1)[0]

            build_dict = {0:'Floor Building Breadth',
                        1:'Floor Building Length',
                        2:'Wall Building Length',
                        3:'Wall Building Breadth',
                        }
            if isprint:
                print('Building Choice is: ',build_dict[building_choice])        
                
            boxes=[]
            if building_choice == floor_building_breadth:
                items=sorted(items, key=lambda x: (x['position'][0], x['position'][1], x['position'][2]))

            elif building_choice == floor_building_length:
                items=sorted(items, key=lambda x: (x['position'][1], x['position'][0], x['position'][2]))

            elif building_choice == wall_building_length:
                items=sorted(items, key=lambda x: (x['position'][2], x['position'][0], x['position'][1]))

            elif building_choice == wall_building_breadth:
                items=sorted(items, key=lambda x: (x['position'][2], x['position'][1], x['position'][0]))

        else:
            items=sorted(items, key=lambda x: (x['position'][2],x['position'][1], x['position'][0]))

        return items
      
      
    def get_coords(self,ldc_len,min_len,_range):
        nhs = []
        h = 0 
        while h<=ldc_len:
            nh = np.random.randint(_range[0],_range[1])
            if h+nh<=ldc_len-min_len:
                h+=nh
    #             print(h)
                nhs.append(h)
            if ldc_len-h<=_range[0]+min_len:
                break
        return nhs

    def get_boxes_train(self,randomseed=True):
        if randomseed==False:
            np.random.seed(self.seed)
            
        len_cuts = np.array(self.get_coords(self.ldc_len,10,[10,50]))
        len_cuts_new = np.copy(len_cuts)
        len_cuts = np.sort(np.append(len_cuts,0))
        len_cuts_new = np.append(len_cuts_new,self.ldc_len)

        wid_cuts = np.array(self.get_coords(self.ldc_wid,10,[10,25]))
        wid_cuts_new = np.copy(wid_cuts)
        wid_cuts = np.sort(np.append(wid_cuts,0))
        wid_cuts_new = np.append(wid_cuts_new,self.ldc_wid)

        ht_cuts = np.array(self.get_coords(self.ldc_ht,10,[10,25]))
        ht_cuts_new = np.copy(ht_cuts)
        ht_cuts = np.sort(np.append(ht_cuts,0))
        ht_cuts_new = np.append(ht_cuts_new,self.ldc_ht)

        lens = len_cuts_new - len_cuts
        wids = wid_cuts_new - wid_cuts
        hts  = ht_cuts_new  - ht_cuts

        floor_building_breadth = 0
        floor_building_length  = 1
        wall_building_length   = 2
        wall_building_breadth  = 3

        building_choice = np.random.randint(0,4,1)[0]

        build_dict = {0:'Floor Building Breadth',
                      1:'Floor Building Length',
                      2:'Wall Building Length',
                      3:'Wall Building Breadth',
                      }

        item_original = {
            'shape':    np.array([self.ldc_len,self.ldc_wid,self.ldc_ht]),  #[length, height, width]
            'position': np.array([0, 0, 0]),   #[x, y, z]
            'rotation':  0,    
        }    
        """
        rotation: 
        [0, 1, 2]->0
        [0, 2, 1]->1
        [1, 0, 2]->2
        [1, 2, 0]->3
        [2, 0, 1]->4
        [2, 1, 0]->5   
        """            
        # initialize item list
        items = []
        
        if building_choice == floor_building_breadth:
            for i in range(len(ht_cuts)):
                for k in range(len(len_cuts)):
                    for j in range(len(wid_cuts)):
                        item=copy.deepcopy(item_original)
                        item['position'][0]=len_cuts[k]
                        item['position'][1]=wid_cuts[j]
                        item['position'][2]=ht_cuts[i]
                        item['shape'][0]=lens[k]
                        item['shape'][1]=wids[j]
                        item['shape'][2]=hts[i]
                        shuffle_i = np.random.randint(0,6)
                        item['rotation']= shuffle_i
                        item['shape'] = get_rotation(item['shape'],shuffle_i)  
                        items.append(item)               
                        


        elif building_choice == floor_building_length:
            for i in range(len(ht_cuts)):
                for j in range(len(wid_cuts)):
                    for k in range(len(len_cuts)):
                        item=copy.deepcopy(item_original)
                        item['position'][0]=len_cuts[k]
                        item['position'][1]=wid_cuts[j]
                        item['position'][2]=ht_cuts[i]
                        item['shape'][0]=lens[k]
                        item['shape'][1]=wids[j]
                        item['shape'][2]=hts[i]
                        shuffle_i = np.random.randint(0,6)
                        item['rotation']= shuffle_i
                        item['shape'] = get_rotation(item['shape'],shuffle_i)  
                        items.append(item)    

        elif building_choice == wall_building_length:
            for j in range(len(wid_cuts)):
                for k in range(len(len_cuts)):
                    for i in range(len(ht_cuts)):
                        item=copy.deepcopy(item_original)
                        item['position'][0]=len_cuts[k]
                        item['position'][1]=wid_cuts[j]
                        item['position'][2]=ht_cuts[i]
                        item['shape'][0]=lens[k]
                        item['shape'][1]=wids[j]
                        item['shape'][2]=hts[i]
                        shuffle_i = np.random.randint(0,6)
                        item['rotation']= shuffle_i
                        item['shape'] = get_rotation(item['shape'],shuffle_i)  
                        items.append(item)        

        elif building_choice == wall_building_breadth:
            for k in range(len(len_cuts)):
                for j in range(len(wid_cuts)):
                    for i in range(len(ht_cuts)):
                        item=copy.deepcopy(item_original)
                        item['position'][0]=len_cuts[k]
                        item['position'][1]=wid_cuts[j]
                        item['position'][2]=ht_cuts[i]
                        item['shape'][0]=lens[k]
                        item['shape'][1]=wids[j]
                        item['shape'][2]=hts[i]
                        shuffle_i = np.random.randint(0,6)
                        item['rotation']= shuffle_i
                        item['shape'] = get_rotation(item['shape'],shuffle_i)  
                        items.append(item)      
        
        return items    





    def get_data_dict(self,train=True,flatten=True):
        ldc = np.zeros((self.ldc_len,self.ldc_wid))
        if train==True:
            boxes = self.get_boxes_train()
        else:
            boxes=self.get_boxes()

        data = []


        for m in range(len(boxes)):

            boxesshape=np.copy(boxes[m]['shape'])
            l = boxes[m]['shape'][0]
            b = boxes[m]['shape'][1]
            h = boxes[m]['shape'][2]
            i = boxes[m]['position'][0]
            j = boxes[m]['position'][1]
            k = boxes[m]['position'][2]
            r = boxes[m]['rotation']

            boxes_shape=get_inverse_rotation(boxesshape,r)

            if flatten:
                ldc_flatten = ldc.flatten()
            else:
                ldc_flatten = np.copy(ldc)
            data.append([ldc_flatten, np.array([l,b,h]), np.array([i,j,k]),r])
            ldc[i:i+boxes_shape[0],j:j+boxes_shape[1]] += boxes_shape[2]
        return data

if __name__ == "__main__":
    import shutil
    if os.path.exists('./Box_data'):
        shutil.rmtree('./Box_data')

    if not os.path.exists('./Box_data'):
        os.makedirs('./Box_data')

    data_maker = BoxMaker()
    # boxes = data_maker.get_boxes()
    # ldc = np.zeros((45,80))
    # ldc_ht = 45
    # for m in range(len(boxes)):
    #     l = boxes[m][0]
    #     b = boxes[m][1]
    #     h = boxes[m][2]
    #     i = boxes[m][3]
    #     j = boxes[m][4]
    #     k = boxes[m][5]
    #     ldc[i:i+b,j:j+l] += h
    #     plt.imshow(ldc,cmap='hot',vmin=0,vmax=ldc_ht)
    #     plt.savefig('Box_data/state_'+str(m)+'.jpg')
    # data_maker.get_data_dict()
    print(np.array(data_maker.get_data_dict()))


