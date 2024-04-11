# -*- coding: utf-8 -*-
"""
Constructing Optimum-path Forest.
"""

import numpy as np

class PriorityQueue:    
    def push(self,node,weight,parent,label,root):
        if hasattr(self, 'queue'):
           self.queue=np.append(self.queue,[[node,weight,parent,label,root]],axis=0)
        else:
            self.queue=np.array([[node,weight,parent,label,root]])
    def pop(self,priority_index):
        if len(self.queue) > 0:
            weights=self.queue[:,priority_index]
            min_weight=np.where(weights == np.amin(weights))
            min_weight_index=min_weight[0][0]
            result=self.queue[min_weight_index,:]
            self.queue=np.delete(self.queue,min_weight_index,axis=0)
        else:
            result=[]
        return result
    def remove(self,node_id):
        row_index=np.where(self.queue[:,0]==node_id)
        self.queue=np.delete(self.queue,row_index,axis=0)
    def empty(self):
        return len(self.queue) == 0    

def find_prototypes(corr_matrix):
    prototypes = list()
    prototypes_temp = list()
    for i in range(0,corr_matrix.shape[0]):
        for j in range(i+1,corr_matrix.shape[0]):
            if corr_matrix[i,j]>0.9:
                if i not in prototypes and i not in prototypes_temp:
                    prototypes.append(i)
                if j not in prototypes_temp:
                    prototypes_temp.append(j)
        print("i =",i)
    return prototypes

def train(corr_matrix,Prototypes):
    Model=np.zeros((corr_matrix.shape[0],5))
    """
    index 0: node id, index 1: cost, index 2: parent, index 3: label, 
    index 4: root
    """
    Model[:,0]=range(np.shape(Model)[0])
    Model[:,1]=10000
    Model[Prototypes,1]=0
    Model[Prototypes,3]= 100 #no matters
    Model[Prototypes,4]=Prototypes
    priority_queue = PriorityQueue()
    for item in Prototypes:
        priority_queue.push(item,0,0,100,item)
    i = 0
    while priority_queue.empty() == False:        
        node,cost,parent,label,root=priority_queue.pop(1)
        #for k in range(np.shape(Z1)[0]):        
        for k in range(corr_matrix.shape[0]):        
            #dst = distance.euclidean(Z1[int(node),1:10], Z1[k,1:10])                  
            dst = 1-corr_matrix[int(node),k]
            max_cost=max(cost,dst)
            if max_cost<Model[k,1]:
                if Model[k,1]!=10000:
                    priority_queue.remove(Model[k,0])
                Model[k,:]=[Model[k,0],max_cost,int(node),label,int(root)]
                priority_queue.push(Model[k,0],Model[k,1],Model[k,2],Model[k,3],Model[k,4])
        i+=1
        print("i =",i)          
    #Model=Model[Model[:,1].argsort()]
    return Model



class PriorityQueueNew:    
    def push(self,node,weight,parent,label,root):
        if hasattr(self, 'queue'):
           self.queue=np.append(self.queue,[[node,weight,parent,label,root]],axis=0)
        else:
            self.queue=np.array([[node,weight,parent,label,root]])
    def pop(self,priority_index):
        if len(self.queue) > 0:
            weights=self.queue[:,priority_index]
            max_weight=np.where(weights == np.amax(weights))
            max_weight_index=max_weight[0][0]
            result=self.queue[max_weight_index,:]
            self.queue=np.delete(self.queue,max_weight_index,axis=0)
        else:
            result=[]
        return result
    def remove(self,node_id):
        row_index=np.where(self.queue[:,0]==node_id)
        self.queue=np.delete(self.queue,row_index,axis=0)
    def empty(self):
        return len(self.queue) == 0   
