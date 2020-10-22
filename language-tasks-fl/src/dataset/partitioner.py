from torch.utils.data import DataLoader, Subset
import numpy as np

class Partition:
    def partitionByClass(self,data):
        lstParts = []
        classes = np.unique(data.targets)
        for y in classes:   
            idx = np.arange(len(data))
            X = Subset(data, idx[data.targets ==y])
            lstParts.append(X)
            
        return lstParts
    def iidParts(self,data,numParts):
        N = len(data)#data.data.shape[0]
        idxs = np.random.permutation(N)
        partIdxs = np.array_split(idxs, numParts)
        lstParts = [Subset(data, partIdx) for partIdx in partIdxs]
        return lstParts
