# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 15:46:31 2020

@author: Ankush
"""

#Boltzmann Machines

#importing the libraries
import numpy as np #because we are using arrays
import pandas as pd
import torch
#importing torch libraries
import torch.nn as nn  #torch module for building neural network
import torch.nn.parallel # for parallel computation
import torch.optim as optim  #for optimization
import torch.utils.data  #some torch tools
from torch.autograd import Variable #for stochasic gradient descent

#Importing the datasets
movies= pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine= 'python', encoding='latin-1')
# 'encoding = 'latin-1'' is used to read special characters in the dataset which is not possible with typical encoding = utf8
#'header =none' means no special row to mention he headings 
users= pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine= 'python', encoding='latin-1')
ratings= pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine= 'python', encoding='latin-1')

#Preparing the training set and the test set
training_set= pd.read_csv('ml-100k/u1.base',delimiter= '/t')#delimiter refers the seperation key which is tab here 
training_set= np.array(training_set, dtype = 'int')
test_set= pd.read_csv('ml-100k/u1.test',delimiter= '/t')
test_set= np.array(test_set,dtype = 'int')

#Getting the number of users and movies
nb_users=int(max(max(training_set[:,0]),max(test_set[:,0])))#finding the maximum number of users by finding the maximum index value which could be in either training or test set
nb_movies=int(max(max(training_set[:,1]),max(test_set[:,1])))#finding the maximum number of movies by finding the maximum index value  which could be in either training or test set


#Converting the data into an array with users in lines and movies in column(list of lists)
#creating a list of list containing all the users and the ratings given by them to respective movies
def convert(data): # a function to create the lis of list
    new_data=[]
    for id_users in range(1,nb_users+1):
        id_movies= data[:,1][data[:,0]==id_users]#an array will be created containing the indexes of movies rated by respective user
#ex.     user 1 - [3,2,4,1,5,3,4,5,2,3,4,2]
#        user 2 - [1,3,3,4,5,2,5,2,3,1,4,3,4]   
        id_ratings= data[:,2][data[:,0]==id_users]#an array will be created containing the indexes of ratings rated by respective user
        ratings=np.zeros(nb_movies)#creating a huge array of zeros with total columns equal to toal no. of movies. 
        ratings[id_movies - 1]= id_ratings #each value in id_ratings will get stored in the rating array at respective indexes
        new_data.append(list(ratings))#the ratings list (which is array) will be placed in another list(list of list) and the process will be iterated for all users
    return new_data
training_set= convert(training_set)#sending the training set as a variable to convert() fn
test_set= convert(test_set)  #sending the test set as variable to convert() fn


#converting the data into torch tensors  
training_set= torch.FloatTensor(training_set) #torch tensors work only with torch arrays so we are converting the training set which is an np array to torch array
test_set= torch.FloatTensor(test_set) #torch tensors work only with torch arrays so we are converting the test set which is an np array to torch array

#converting the ratings into binary ratings 1(liked) or 0 (unliked)
training_set[training_set== 0]=-1 #it interprets that we are changing all the 0s values in training set with -1. simple 
training_set[training_set== 1]=0 #for line 62 and 63 we could have used "1 or 2" to save writing two lines but this format is not supported by torch.  
training_set[training_set== 2]=0
training_set[training_set>=3]=1
test_set[test_set== 0]=-1
test_set[test_set== 1]=0
test_set[test_set== 2]=0
test_set[test_set>=3]=1

#Creating the architechture of neural network
class RBM:
    def __init__(self,nv,nh):
        self.W= torch.randn(nh,nv)#creating a torch tensor(which is an matrix)to initialize the weights
        self.a= torch.randn(1,nh)#creating a torch tensor(which is an matrix)to initialize the bias for hidden nodes.
        # "1 and nh" represents 2 dimensions which is not neccessary but torch tensors accept only multidimensional arrays 
        self.b= torch.randn(1,nv)#creating a torch tensor(which is an matrix)to initialize the bias for input nodes
        #"self.a or self.b" a nd b represents 'parameters' of object 'self'
        
    def sample_h(self,x):#x is visible nodes input values
        wx= torch.mm(x, self.W.t()) # get product of [input neurons]•[Weights.transpose()]. Thats the way in torch to multiply vectors
        #   transpose is needed to rotate the weights matrix for multiplications
        #   the same orientation as the input values
        activation= wx + self.a.expand_as(wx)# create activation value by adding the [hidden node] bias to the weighted input (wx)
        #   where hidden node bias is expanded to the shape of the weighted inputs. ".expand_as" is making sure that all that values of
        #   are multiplied by correspondind values of a(which is bias)
        p_h_given_v= torch.sigmoid(activation) # caluclate probability of hidden node activation for given visible node weight
        # (dont know what it is )Bernoulli sampling involves generating a random number between 0 and 1 then 
        # returning a 1 if the rand <= activation_probability (to activate the neuron (100*prob)% of the time)
        return ph_given_v,torch.bernaulli(ph_given_v)
                                                            
    def sample_v(self,y):#y is visible nodes input values
        wy= torch.mm(y, self.W)#we dont need transpose here because the dimensions are same .
        #note-You always transpose whenever the matrix dimensions dont agree. 
        #In Matrix multiplication, let's say you want to multiply two matrices of size (4x3) and (2x3). Then in order to do that you have to transpose (2x3) so it can be a (3x2). 
        #Then the dimensions agree, so it will be (4x3) x (3x2), and you will get a (4x2) output matrix
        activation= wy+ self.b.expand_as(wy)#calculating activation values.
        p_v_given_h= torch.sigmoid(activation)# # calculate probability of  activation of visible node for given hidden node weights
        return p_v_given_h,torch.bernaulli(p_v_given_h)#you get a value between 0 and 1 which is fed into torch bernaulli that decides whether the node will be activated or not.
        #lets say the threshold value is 0.3 , then it means 30% chance that the value is 1 and 70% that the value is 0. Now if the sigmoid value is less than equal to 0.30, then
        #the return value is 1 , else the return value is 0.the threshhold value is decided by the tensor .(how? need more reserch)initially the TV is 0.5 but changes with continuous feed of data.
        # ex--->> m = Bernoulli(torch.tensor([0.3]))
        # m.sample()  30% chance 1; 70% chance 0
        #tensor([ 0.])
        #and you wil  get 0.3 from sigmoid fn
        #-----------------------------------
        #the below fn is for contrastive divergence. It is mainly used to minimize the energy (or loss) just like backpropagation.BP cant handle big calculation ,so we are using CD which also has gibbs sampling
    def train(self,v0,vk,ph0,phk):#v0 is the initial visible node vector[containing values of all visible nodes in an array],vk is the final or kth visible node vector,
        #ph0 is the probability of initial hidden node vector[initial hidden nodes created from the input vector],phk is the probability of final or kth hidden node vector[kth hidden node created from the kth input vector]          
        self.W += (torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)).t()#torch.mm is used for matrix multiplication
        self.b+=torch.sum((v0-vk), 0)#adjusting bias for the hidden nodes 
        self.a+=torch.sum((ph0-phk), 0)#adjusting bias for the input nodes 
    def predict(self, x): # x: visible nodes
       _,h = self.sample_h(x)
       _,v = self.sample_v(h)
       return v       
nv=len(training_set[0])#total no. of visible nodes which is equal to total no. of movies.
nh=100 # total hidden nodes are decided by us. It can be tuned / changed.
batch_size=100 #same as backpropagation. After each batch the contrastive divergence will be applied to update the weights.
rbm=RBM(nh,nv) #creating an object of RBM fn with nh and nv as arguments.
#we can conclude that for each user we will have 1682 input nodes(no. of movies) and 100 hidden nodes. After 100 users(rows) we will perform CD(earlier we used to do Stochastic gradient descent) to update the weights    

#Training the RBM
nb_epoch=10#same as previous. note-you should know how data is arrange in training and test set(what represents rows and what represents coloumns)
for epoch in range(1,nb_epoch+1):
    train_loss=0 #it is loss variable. Initially it is zero because there is no loss detected.
    s=0.#it is a counter variable.It is used for normalization of loss variable which is train_loss. We are basically gonna divide the train_loss with this counter variable. "dot" is used to make it a float value.
    for id_user in range(0,nb_users-batch_size,batch_size):#we are creating a for loop where all the values in the batch are updated at once(it means a single loop is performing for all the values of batch at once)
        #the above parameters represents(0, 943-100=843, 100). It means 0-99,100-199,200-299....800-900. The values will exceed 843 even though the last value of range is 843.Thats how it works .
        #Also by using this method, we are losing last 43 values.                                                                                                                                                                 
        v0=training_set[id_user: id_user+batch_size]# I believe that it is a multidimensional matrix containing all the movie ratings of each and every single batch users. [1,0,0,1,1,1,0,0,1,0,0,1, 1  . It doesnt matter how the ratings are represented for each user.(multidimensional or vertical vectors) 
        #because calculationas will be done on each user SEPERATELY with common or SAME set of hidden nodes throughout the procedure.
                                                                                                                                                                 #            0,0,1,1,1,0,1,1,0,0,0,1,1
                                                                                                                                                                  #       last_user_batch_ratings......]
        vk=training_set[id_user: id_user+batch_size]#it is the last vector of the gibbs sampling.initially vo and vk are same because because k in gibbs sampling is 0
        ph0,_= rbm.sample_h(v0)# used for initial hidden probabilities that is calculated by initial input nodes(significance?) 
        for k in range(10):#below we are using gibbs sampling.for that we are using 'for' loop.#initially vk is v0. but k will change with loop. the reason loop is effective and changing the values of v0(which earlier i believed remained constant) is be
            #because during the loop vk is getting UPDATED (line 136)since vk is getting constructed from sample_h fn(which causes the fn ) and that updated vk is used as an input for hk. Therefore, the updated hk will go as an input to the next line(line 136) and the process is repeated untill the loop ends. 
            #after this, we get an updated vk value.that updated vk value will be used to update the phk and to update the weights and bias during training. it will be continued for all epochs.next line cont.
            _,hk=rbm.sample_h(vk)#just remember, vk and hk values will never be same or constant during the loop since they are constructed value . the functions(here sample_h and sample_v) will never construct same values..next line cont.
            _,vk=rbm.sample_h(hk)
            vk[v0<0]=v0[v0<0] #What is this?
        phk,_=rbm.sample_h(vk)#calculating te final hidden probabibilities with vk which is the updated one                                  
        rbm.train(v0,vk,ph0,phk)#training the rbm by putting the required parameters
        train_loss+=torch.mean(torch.abs[v0[v0>=0]-vk[v0>=0]])#calculating the loss for each epoch by the difference of absolute values. we are using 'average distance' to find the error. Another method-RMSE

        s+=1.#it is used to update i.e. increase the counter.note->At the end of each epoch, the losses are normalized by the same number, however when training happens, we see the updated loss w.r.t the number of samples that have been trained so far
    print('epoch: '+str(epoch)+'loss: '+str(train_loss/s))#printing the epoch and loss value for each epoch.remember loss is calculated after normalization(thats why we are dividing by s)
#at the end of the procedure ,vk might change during training depending upon the users or rows we feed in at a time.(obvious)
#IN the end , we need a trained phk(hidden nodes) that can predict the new values depending upon its training(how good weights and bias have been made and updated).  
        
#Training the RBM
test_loss=0
s=0.
for id_user in range(nb_users):#
    v=training_set[id_user:id_user+1]#v and vt are supposed to be equal. Then why are we taking them from different sets? significance ?
    vt=test_set[id_user:id_user+1]
    if len(vt[vt>=0])>0:#we are using oly one step for contrastive divergence. therefore, no for loop. the 'if' is notvery neccessary. cont.
        _,h= rbm.sample_h(v)#cont.  it just tells that the total values/ratings of movies that were given by user are more than 0.(since total no. rated movies is very small.That no. should be greater than zero)
        _,v= rbm.sample_v(h)#v represents the predicted input values. it will be compared with vt
        test_loss+= torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))#we are using 'average distance' to find the error. Another method-RMSE
print('test loss : '+str(test_loss/s))
#we didnt use ph0,phk because this is r=test_set. we are trying to directly get the predicted input nodes in one step. That is our answer    
        
#---------------------------------------------------------------------------------------------------------------------

#1) Created a predict method in the RBM class. look in the rbm class


#2) Take any user by the test dataset and Take the whole list of movies of that user and convert to PyTorch
user_id = 23
user_input = Variable(test_set[id_user-1]).unsqueeze(0)

#3) Make the prediction using this input

output = rbm.predict(user_input)
output = output.data.numpy()

#4) Stack the inputs and outputs as one numpy array
input_output = np.vstack([user_input, output])    
        
#Questions -       
#what is the relationship between training and test set? how is training code has any significance to the test code? 
#we are not even using any trained object or variable that will the test code to predict the new values        
        
#how are we dealing with -1 here?the prediction code(after line 156) did a fantastic job predicting the input values and comparing it with the intial one.cont.
#with -1 ratings replaced with '0' or '1' but the question remains how it is dealing with -1 and replacing it with 1 or 0 as desired by us? Which part of code deals with this?    
   
        
        
        
        





