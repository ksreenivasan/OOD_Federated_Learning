import torch.nn 
import torch.optim as optim
from models.lenet import *
from models.lenet import LeNet5
from torch.utils.data import DataLoader
import logging
import os
from globalUtils import *
import globalUtils

import torch
#from torchtext import data
#from torchtext import datasets
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .generic_model_training import GenericModelTraining

class TextBCModelTraining(GenericModelTraining):
    
    def __init__(self, config,lr=None, isAttacker=False, loadFromCkpt=False, trainData=None,
                       testData=None,workerId=0,activeWorkersId=None):
        super().__init__(config,lr,isAttacker,loadFromCkpt,trainData,testData,workerId,activeWorkersId)
            
   
    def trainOneEpoch(self,epoch,w0_vec=None):
        clip = 5
        self.model.train()
        hidden =  self.model.initHidden(self.trainConfig['batchSize'])
        epochLoss = 0
        for batchIdx, (data, target) in enumerate(self.trainLoader):
            if(len(data) < self.trainConfig['batchSize']):
                self.logger.info('ignore batch due to small size = {}'.format(len(data)))
                continue
                
            data, target = data.to(self.device), target.to(self.device)
            hidden = tuple([each.data for each in hidden])
            self.optim.zero_grad()   # set gradient to 0
            #hidden = self.repackage_hidden(hidden) 
            
            output,hidden       = self.model(data,hidden)
            #print(output.shape,target.shape)
            #print(output)
            loss         = self.model.criterion(output.squeeze(), target.float())
            loss.backward()    # compute gradient
            
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip) 
            #print(self.trainConfig)
            if(self.trainConfig['method']=='pgd'):     
                eps = self.trainConfig['epsilon']  
                # make sure you project on last iteration otherwise, high LR pushes you really far
                if (batchIdx%self.trainConfig['projectFrequency'] == 0 or batchIdx == len(self.trainLoader)-1):
                    self.projectToL2Ball(w0_vec,eps)

            if batchIdx%20 == 0:
                self.logger.info('Worker: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(self.workerId,
                                epoch, batchIdx * len(data), len(self.trainLoader.dataset),
                                100. * batchIdx / len(self.trainLoader), loss.item()))
            self.optim.step()
            #self.hidden = hidden
            epochLoss += loss.item()
        #self.model.eval()
        #self.logger.info("Accuracy of model {}".format(self.workerId))
        #currTestLoss, curTestAcc = self.validateModel()
        #return currTestLoss, curTestAcc
        #lss,acc_bf_scale = self.validate_model(logger)
        return epochLoss,0,0
     
   
    def validateModel(self,model=None,dataLoader=None):
        if(model is None):
            model = self.model
        if(dataLoader is None):
            dataLoader = self.testLoader
            
        model.eval()
        testLoss = 0 
        correct = 0 
        hidden =  self.model.initHidden(self.trainConfig['testBatchSize'])
        with torch.no_grad():
            for batchIdx, (data, target) in enumerate(dataLoader):
                data, target = data.to(self.device), target.to(self.device)
                hidden       = self.repackage_hidden(hidden) 
                output,hidden       = model(data,hidden)
                testLoss    += self.model.criterion(output.squeeze(), target.float()).item()
               
                #pred = torch.max(output, 1)[1]
                #pred = torch.round(output.squeeze())
                #print(pred)
                #correct += (pred == target).float().sum()
        
                pred         = torch.round(output.squeeze()) #output.max(1, keepdim=True)[1]
                correct     += pred.eq(target.view_as(pred)).sum().item()
        
        testLoss /= len(dataLoader)
        testAcc   =  100. * correct / len(dataLoader.dataset)
        #self.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #                    testLoss, correct, len(dataLoader.dataset), testAcc))
        return testLoss, testAcc 
 
    def repackage_hidden(self,h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
    
    #def trainOneAdversarialEpoch(self):
        
        
