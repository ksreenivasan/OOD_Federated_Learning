from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
from .partitioner import Partition

class MNISTData:
    def __init__(self,dataPath=None):
        self.dataDir = dataPath
        
        
    def buildDataset(self):
        
        self.trainData = MNIST(self.dataDir, download=True,
                                transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
        
        self.testData  = MNIST(self.dataDir, train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((32, 32)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    
    def getTrainDataForUser(self,userId):
        return self.lstParts[userId]
                
    def partitionTrainData(self,partitionType,numParts):
        partitioner = Partition()

        if(partitionType=='iid'):
            self.lstParts = partitioner.iidParts(self.trainData, numParts)
        elif(partitionType=='non-iid'):
            self.lstParts = partitioner.niidParts(self.trainData,numParts)
        else:
            raise('{} partitioning not defined for this dataset'.format(partitionType))


