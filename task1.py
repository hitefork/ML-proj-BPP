'''
CNN with history of 4 boxes as states
'''
import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import os
from make_data import BoxMaker
from model import StochasticPolicyCNN,StochasticPolicy
from config_task1 import args
from make_data import get_inverse_rotation

import matplotlib.pyplot as plt
# configure CUDA availability


use_cuda = torch.cuda.is_available()

device   = torch.device("cuda:0" if use_cuda else "cpu")

max_ldc_x  = 1
max_ldc_y =1

def getFeasibility(ldc,x,y,l,b,h,r):
    feasible = False
    rotated_shape=get_inverse_rotation(np.array([l,b,h]),r)
    # print(ldc[x:x+rotated_shape[0],y:y+rotated_shape[1]])
    if len(np.unique(ldc[x:x+rotated_shape[0],y:y+rotated_shape[1]])) == 1 and np.all(ldc[x:x+rotated_shape[0],y:y+rotated_shape[1]]+rotated_shape[2]<=100):
        feasible=True
    return feasible

class ReplayBuffer(object):
    def __init__(self, max_size=1e4):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r = [], [], [], []

        for i in ind:
            X, Y, U, R = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(R)      
            
        return np.array(x), np.array(y), np.array(u),np.array(r)
    
class BehaviouralCloning():
    def __init__(self,args,ldc_len=100,ldc_wid=100,ldc_ht=100,search_range=6,name="StochasticPolicyCNN_train"):
        self.ldc_len = ldc_len
        self.ldc_wid = ldc_wid
        self.ldc_ht  = ldc_ht
        self.num_actions = 3
        self.input_size = self.ldc_len*self.ldc_wid
        self.data_maker = BoxMaker(self.ldc_ht,self.ldc_wid,self.ldc_len)
        self.policy = StochasticPolicyCNN().to(device)
        self.search_range=search_range # will search in +-search_range
        self.search = np.arange(0,search_range,1)
        self.neg_search = -np.arange(1,search_range,1)
        self.search_arr = np.append(self.search,self.neg_search)
        self.name=name
        if args.tensorboard:
            print('Init tensorboardX')
            self.writer = SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))


    def shift_action(self,action,rotation):
        x = action[:,0]
        y = action[:,1]
        r = rotation
        x = x*self.ldc_len/2 +self.ldc_len/2
        y = y*self.ldc_wid/2 +self.ldc_wid/2
        return x,y,r

    def train(self):
        optimizer = optim.Adam(self.policy.parameters(),
                                     lr=args.lr)
        start_episode = 0

        buff = ReplayBuffer(1e4)

        if args.load_path!=None:
            if not use_cuda:
                checkpoint = torch.load(args.load_path,map_location='cpu')
            else:
                checkpoint = torch.load(args.load_path)
                print("load: "+args.load_path)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode']
            loss = checkpoint['loss']
            self.policy.train()

        for episodes in tqdm(range(args.episodes)):
            self.data_maker=BoxMaker(self.ldc_ht,self.ldc_wid,self.ldc_len)
            data = self.data_maker.get_data_dict(flatten=False)
            data = np.array(data)
            state = np.zeros((4,self.ldc_len,self.ldc_wid))
            dim   = np.zeros((12))
            for i in range(len(data)):
                state = np.roll(state,axis=0,shift=1)
                state[0,:,:] = data[i][0]
                dim = np.roll(dim,shift=3)
                dim[:3] = data[i][1]
                action = data[i][2][:2]
                rotation=data[i][3]
                buff.add([state,dim,action,rotation])
                
            self.search = np.arange(0,self.search_range,1)
            self.neg_search = -np.arange(1,self.search_range,1)
            self.search_arr = np.append(self.search,self.neg_search)
            if len(buff.storage) >= args.batch_size:
                state_feed, dim_feed, action_feed,rotation_feed = buff.sample(args.batch_size)
                state_feed   = torch.FloatTensor(state_feed)/self.ldc_ht
                dim_feed     = torch.FloatTensor(dim_feed)/self.ldc_ht

                action_feed  = torch.from_numpy(action_feed)
                rotation_feed  = torch.from_numpy(rotation_feed)
                 
                if use_cuda:
                    state_feed   = state_feed.to(device)
                    dim_feed     = dim_feed.to(device)
                    action_feed  = action_feed.to(device)
                    rotation_feed  = rotation_feed.to(device)


                a,m,s,r   = self.policy.sample(state_feed.float(),dim_feed.float())
                x,y,temp_rotation     = self.shift_action(a,r)
                
                pred = torch.cat([x.unsqueeze(1),y.unsqueeze(1)],dim=1)
                rotation_pred=torch.Tensor(temp_rotation)

                optimizer.zero_grad()
                loss_action = F.mse_loss(pred,action_feed.float())
                loss_rotation = F.mse_loss(rotation_pred.squeeze(1),rotation_feed.float())
                total_loss=loss_action+loss_rotation
                total_loss.backward()
                
                
                # print(loss.item())
                if args.tensorboard:
                    self.writer.add_scalar('Loss',total_loss.item(),episodes+start_episode)
                optimizer.step()

            if episodes % 5000 == 0 and episodes !=0:
                # print('Saving model...')
                if not os.path.exists(args.save_path+self.name): #判断所在目录下是否有该文件名的文件夹
                    os.mkdir(args.save_path+self.name) #创建多级目录用mkdirs，单击目录mkdir
                torch.save({
                            'episode': episodes,
                            'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': total_loss,
                            }, args.save_path+self.name+"/"+str(episodes)+".pt")
        self.writer.close()
        
    def getStabilityScore(self,i, j , ldc, dimn,currldc_x=0, currldc_y=0,current_r=0):
    #     level = ldc[i,j]
        rotated_shape=get_inverse_rotation(np.array(dimn),current_r)
        h = rotated_shape[2]
        feasible = False
        found_flat = found_not_flat = 0
        if  (j >= self.ldc_len*currldc_x) and (j+rotated_shape[0] <= self.ldc_len*(currldc_x+1)) and\
            (i >= self.ldc_wid*currldc_y) and (i+rotated_shape[1] <= self.ldc_wid*(currldc_y+1)):
            level = ldc[i,j]
            if level + h <= self.ldc_ht: 
                feasible = True
                # --------------------------------------------------- Flat position
                if len(np.unique(ldc[i:i+rotated_shape[1], j:j+rotated_shape[0]])) == 1:
                    stab_score = 1
                    found_flat = 1
                # ---------------------------------------------------- Non-Flat position
                if not found_flat:
                    corners =  [ldc[i,j], ldc[i+rotated_shape[1]-1,j], ldc[i,j+rotated_shape[0]-1], ldc[i+rotated_shape[1]-1, j+rotated_shape[0]-1]]
                    if (np.max(corners) == np.min(corners)) and (np.max(corners) == np.max(ldc[i:i+rotated_shape[1],j:j+rotated_shape[0]])):
                        stab_score = - np.sum(np.max(corners)-ldc[i:i+rotated_shape[1],j:j+rotated_shape[0]])/(rotated_shape[0]*rotated_shape[1]*self.ldc_ht)
                        found_not_flat = 1


        if (found_flat) or (found_not_flat):
            minj = np.max((self.ldc_len*currldc_x,j-1))
            maxj = np.min((self.ldc_len*(currldc_x+1),j+rotated_shape[0]))
            mini = np.max((self.ldc_wid*(currldc_y),i-1))
            maxi = np.min((self.ldc_wid*(currldc_y+1),i+rotated_shape[1]))

            # Border for the upper edge
            if i==currldc_y*self.ldc_wid: 
                upper_border = (self.ldc_ht - 1 + np.ones_like(ldc[mini,j:(j+int(rotated_shape[0]))])).tolist()
            else: 
                upper_border  = ldc[mini,j:(j+int(rotated_shape[0]))].tolist()
            # Stability for the upper edge
            unique_ht = np.unique(upper_border)
            if len(unique_ht) == 1:
                stab_score += 0.5
                if unique_ht[0] == level: stab_score -= 2
                elif unique_ht[0] == self.ldc_ht: stab_score += 1.5
                else:
                    sscore = 1.-abs(unique_ht[0]-(level+h))/self.ldc_ht
                    if (unique_ht[0]>level): stab_score += 1.5*sscore
                    else:                    stab_score += 0.75*sscore
            else:
                stab_score += 0.25*(1.-len(unique_ht)/h)
                stab_score += 0.25*(1.-sum(abs(ht-(level+h)) for ht in unique_ht)/(h*len(unique_ht)))
                stab_score += 0.50*sum(ht!=level for ht in unique_ht)/len(unique_ht)
            #border.extend(upper_border)
            del upper_border

            # Border for the left edge
            if j==currldc_x*self.ldc_len:
                left_border = (self.ldc_ht - 1 + np.ones_like(ldc[i:(i+int(rotated_shape[1])),minj])).tolist()
            else: 
                left_border = ldc[i:(i+int(rotated_shape[1])),minj].tolist()
            # Stability for the left edge
            unique_ht = np.unique(left_border)
            if len(unique_ht) == 1:
                stab_score += 0.5
                if unique_ht[0] == level: stab_score -= 2
                elif unique_ht[0] == self.ldc_ht: stab_score += 1.5
                else:
                    sscore = 1.-abs(unique_ht[0]-(level+h))/self.ldc_ht
                    if (unique_ht[0]>level): stab_score += 1.5*sscore
                    else:                    stab_score += 0.75*sscore
            else:
                stab_score += 0.25*(1.-len(unique_ht)/h)
                stab_score += 0.25*(1.-sum(abs(ht-(level+h)) for ht in unique_ht)/(h*len(unique_ht)))
                stab_score += 0.50*sum(ht!=level for ht in unique_ht)/len(unique_ht)
            #border.extend(left_border)
            del left_border

            # Border for the lower edge
            if (i+rotated_shape[1] < self.ldc_wid*(currldc_y+1)): lower_border = ldc[maxi,j:(j+int(rotated_shape[0]))].tolist()
            else: lower_border = (self.ldc_ht - 1 + np.ones_like(ldc[maxi-1,j:(j+int(rotated_shape[0]))])).tolist()
            # Stability for the lower edge
            unique_ht = np.unique(lower_border)
            if len(unique_ht) == 1:
                stab_score += 0.5
                if lower_border[0] == level: stab_score -= 2
                elif lower_border[0] == self.ldc_ht: stab_score += 1.5
                else:
                    sscore = 1.-abs(unique_ht[0]-(level+h))/self.ldc_ht
                    if (unique_ht[0]>level): stab_score += 1.5*sscore
                    else:                    stab_score += 0.75*sscore
            else:
                stab_score += 0.25*(1.-len(unique_ht)/h)
                stab_score += 0.25*(1.-sum(abs(ht-(level+h)) for ht in unique_ht)/(h*len(unique_ht)))
                stab_score += 0.50*sum(ht!=level for ht in unique_ht)/len(unique_ht)
            #border.extend(lower_border)
            del lower_border

            # Border for the right edge
            if (j+rotated_shape[0] < (currldc_x+1)*self.ldc_len): right_border = ldc[i:(i+int(rotated_shape[1])),maxj].tolist()
            else: 
                right_border = (self.ldc_ht - 1 + np.ones_like(ldc[i:(i+int(rotated_shape[1])),maxj-1])).tolist()
            # Stability for the right edge
            unique_ht = np.unique(right_border)
            if len(unique_ht) == 1:
                stab_score += 0.5
                if right_border[0] == level: 
                    stab_score -= 2
                elif right_border[0] == self.ldc_ht: 
                    stab_score += 1.5
                else:
                    sscore = 1.-abs(unique_ht[0]-(level+h))/self.ldc_ht
                    if (unique_ht[0]>level): 
                        stab_score += 1.5*sscore
                    else:                    
                        stab_score += 0.75*sscore
            else:
                stab_score += 0.25*(1.-len(unique_ht)/h)
                stab_score += 0.25*(1.-sum(abs(ht-(level+h)) for ht in unique_ht)/(h*len(unique_ht)))
                stab_score += 0.50*sum(ht!=level for ht in unique_ht)/len(unique_ht)
            #border.extend(right_border)
            del right_border
            
            
            # Check the upper edge for continuity
            if i == currldc_y*self.ldc_wid: stab_score += 0.02
            else:
                # In the upper-left corner
                if (j == currldc_x*self.ldc_len) :
                    stab_score += 0.01
                # In the upper-right corner
                if ((j+rotated_shape[0]) == (currldc_x+1)*self.ldc_len) :
                    stab_score += 0.01
            # Check the lower edge for continuity
            if i+rotated_shape[1] == self.ldc_wid*(currldc_y+1): 
                stab_score += 0.02
            else:
                # In the lower-left corner
                if (j == currldc_x*self.ldc_len) :
                    stab_score += 0.01
                # In the lower-right corner
                if ((j+rotated_shape[0]) == (currldc_x+1)*self.ldc_len) :
                    stab_score += 0.01
            # Check the left edge for continuity
            if j == currldc_x*self.ldc_len: 
                stab_score += 0.02
            else:
                # In the upper-left corner
                if (i == currldc_y*self.ldc_wid): 
                    stab_score += 0.01
                # In the lower-left corner
                if (i+rotated_shape[1] == self.ldc_wid*(currldc_y+1)): 
                    stab_score += 0.01
            # Check the right edge for continuity 
            if j+rotated_shape[0] == (currldc_x+1)*self.ldc_len: 
                stab_score += 0.02
            else:
                # In the upper-left corner
                if (i == currldc_y*self.ldc_wid) : 
                    stab_score += 0.01
                # In the lower-left corner
                if (i+rotated_shape[1] == self.ldc_wid*(currldc_y+1)): 
                    stab_score += 0.01 
                
            stab_score -= currldc_x/max_ldc_x + currldc_y/max_ldc_y
            stab_score -= 0.05*(i/((currldc_y+1)*self.ldc_wid) + j/((currldc_x+1)*self.ldc_len))
            stab_score -= level / self.ldc_ht
            #stab_score += (rotated_shape[0]*rotated_shape[1])/(self.ldc_len*self.ldc_wid)
        else:
            stab_score = -10
            
        return stab_score


                    
    def get_pos(self,state,dims):
        #state=(4,100,100)
        #dim=(12)
        
        feed_state = torch.FloatTensor(state)/self.ldc_ht  #4，100，100
        feed_state = feed_state.unsqueeze(0)  #1，4，100，100

        feed_dim = torch.FloatTensor(dims)/self.ldc_ht #12
        feed_dim = feed_dim.unsqueeze(0) #1，12
        
        if use_cuda:
            feed_state   = feed_state.to(device)
            feed_dim     = feed_dim.to(device)

        cur_dim = np.array(dims[:3]).astype(np.uint16)
        a,m,s,r   = self.policy.sample(feed_state.float(),feed_dim.float())
        x,y,temp_rotation     = self.shift_action(a,r)

        temp_rotation=int(temp_rotation[0])
        x = int(x.cpu()[0])
        y = int(y.cpu()[0])
        score = self.getStabilityScore(x,y , state[0,:,:], dimn = cur_dim, currldc_x=0, currldc_y=0,current_r=temp_rotation)
        result_rotation=temp_rotation
        if score <= 0:
            x_nw = x
            y_nw = y
            for x_s in self.search_arr:
                for y_s in self.search_arr:
                    for r_s in range(0,6):
                        search_score = self.getStabilityScore(x_nw+x_s,y_nw+y_s , state[0,:,:], dimn = cur_dim, currldc_x=0, currldc_y=0,current_r=r_s)
                        if search_score > score:
                            score = search_score
                            x = x_nw+x_s
                            y = y_nw+y_s
                            result_rotation=r_s
                            
        return x,y,score,result_rotation
    
    
    def step(self,state,action,dims,r):
        rotated_shape=get_inverse_rotation(dims,r)
        l,b,h = rotated_shape
        x,y = action
        ldc = np.copy(state[0,:,:])
        state = np.roll(state,axis=0,shift=1)
        state[0,:,:] = ldc
        state[0,x:x+l,y:y+b] += h
        return state
    
    
    def evaluate(self):


        if args.load_path!=None:
            if not use_cuda:
                checkpoint = torch.load(args.load_path,map_location='cpu')
            else:
                checkpoint = torch.load(args.load_path)
                print("load: "+args.load_path)
            self.policy.load_state_dict(checkpoint['model_state_dict'])

        self.policy.eval()
        dim   = np.zeros((12))
        self.data_maker=BoxMaker(self.ldc_ht,self.ldc_wid,self.ldc_len)
        data = self.data_maker.get_data_dict(train=False,flatten=False)
        dims=[]


        for i in range(len(data)):
            dim = np.roll(dim,shift=3)
            dim[:3] = data[i][1]
            dims.append(dim)


        tot_vol = 0
        state = np.zeros((4,self.ldc_len,self.ldc_wid))
        walle_vol = 0
        walle_score=0
        packman_num=0
        self.search_space=[]
        for i in range(len(dims)):
            packman = 0
            cur_dim = np.array(dims[i][:3]).astype(np.uint32)


            
            feed_state = torch.FloatTensor(state)/self.ldc_ht  #4，100，100
            feed_state = feed_state.unsqueeze(0)  #1，4，100，100

            feed_dim = torch.FloatTensor(dims[i])/self.ldc_ht #12
            feed_dim = feed_dim.unsqueeze(0) #1，12
            
            if use_cuda:
                feed_state   = feed_state.to(device)
                feed_dim     = feed_dim.to(device)

            cur_dim = np.array(dims[i][:3]).astype(np.uint16)
            a,m,s,r   = self.policy.sample(feed_state.float(),feed_dim.float())
            x,y,temp_rotation     = self.shift_action(a,r)
            l,b,h = cur_dim
        #     print(score)
            temp_rotation=int(temp_rotation[0])
            x = int(x.cpu()[0])
            y = int(y.cpu()[0])
            
            
            # print(cur_dim)
            # print(x,y,temp_rotation)
            feasible = getFeasibility(state[0],x,y,l,b,h,temp_rotation)

            # scale = 1
            # flag = 0
            # rotate = 0
            # while(not feasible and rotate < 6):
            #     upperboundx = min(x+scale,100)
            #     lowerboundx = max(x-scale,0)
            #     upperboundy = min(y+scale,100)
            #     lowerboundy = max(y-scale,0)
                
            #     if(not feasible):
            #         x,y = lowerboundx,lowerboundy
            #         while(y <= upperboundy and not flag):
            #             #temp_rotation = np.random.randint(3)
            #             feasible = getFeasibility(state[0],x,y,l,b,h,temp_rotation)
            #             if(feasible):
            #                 break
            #             y = y+1
                
            #     if(not feasible):
            #         x,y = upperboundx,lowerboundy
            #         while(y <= upperboundy and not flag):
            #             #temp_rotation = np.random.randint(3)
            #             feasible = getFeasibility(state[0],x,y,l,b,h,temp_rotation)
            #             if(feasible):
            #                 break
            #             y = y+1

            #     if(not feasible):
            #         x,y = lowerboundx+1,lowerboundy
            #         while(x < upperboundy and not flag):
            #             #temp_rotation = np.random.randint(3)
            #             feasible = getFeasibility(state[0],x,y,l,b,h,temp_rotation)
            #             if(feasible):
            #                 break
            #             x = x+1

            #     if(not feasible):
            #         x,y = lowerboundx+1,upperboundy
            #         while(x < upperboundy and not flag):
            #             #temp_rotation = np.random.randint(3)
            #             feasible = getFeasibility(state[0],x,y,l,b,h,temp_rotation)
            #             if(feasible):
            #                 break
            #             x = x+1

            #     if(not feasible):
            #         scale = scale+1
            #         if(scale > 80):
            #             scale = 1
            #             temp_rotation = (temp_rotation+1)%6
            #             rotate = rotate+1

            j = 0
            k = 0
            rotate = 0
            while(not feasible and j<100):
                x,y,temp_rotation = j,k,rotate
                #temp_rotation = np.random.randint(3)
                feasible = getFeasibility(state[0],x,y,l,b,h,temp_rotation)
                rotate = rotate+1
                if(rotate >= 6):
                    rotate = 0
                    k = k+1
                if(k >= 100):
                    k = 0
                    j = j+1

            if feasible:
                state = self.step(state,[x,y],cur_dim,temp_rotation)
                tot_vol += cur_dim[0]*cur_dim[1]*cur_dim[2]
                packman_num+=1


        print(tot_vol/(self.ldc_ht*self.ldc_len*self.ldc_wid)*100)
        self.show(state[0,:,:])





    def show(self,a):
        plt.imshow(a,cmap='hot',vmin=0,vmax=self.ldc_ht)
        plt.colorbar()
        plt.savefig('Box_data/task1.jpg')
        

if __name__ == "__main__":
    if not os.path.exists('./Models'):
        os.makedirs('./Models')
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    BC = BehaviouralCloning(args,name="StochasticPolicyCNN_task1")
    # BC.train()
    BC.evaluate()


        
