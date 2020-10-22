from dataset.mnist import MNISTData
from dataset.imdb_data import IMDBData
from dataset.femnist import FEMNISTData
from dataset.sentiment140_data import TwitterSentiment140Data 
def loadDataset(conf):
    datasetName = conf['dataset']
    dataPath    = conf['dataPath']
    if(datasetName == 'mnist'):
        return MNISTData(dataPath)
    if(datasetName == 'femnist'):
        return FEMNISTData(dataPath)
    if(datasetName == 'imdb'):
        return  IMDBData(dataPath)
    if(datasetName == 'sent140'):
        return TwitterSentiment140Data(dataPath) 
    else:
        print('Datset {} Not Defined'.format(datasetName))
        return None

