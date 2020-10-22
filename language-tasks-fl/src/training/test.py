import sys
sys.path.append('../')
from globalUtils import *
#from training.text_bc_training import ModelTraining
from dataset.datasets import loadDataset
from model_training_factory import *
#from dataset import reviewsData
#from dataset import twitter_data
#from dataset import sentiment140_data

if __name__ == "__main__":
    
    #args = add_fit_args(argparse.ArgumentParser(description="Federated Setup"))
    seed(42)
    workerId = 0
    stdoutFlag = True
    logger = getLogger("fl.log", stdoutFlag, logging.INFO) 
    
    configFile = sys.argv[1]
    print("loading conf from {}".format(configFile))
    config = loadConfig(configFile)
    curDataset = loadDataset(config)
    curDataset.buildDataset(conf=config)
    
    #curDataset = reviewsData.ReviewData(reviewsConfig['dataPath'])
    #curDataset.buildDataset()
    #config = sent140Config
    #curDataset  = sentiment140_data.TwitterSentiment140Data(config['dataPath'])
    #curDataset.buildDataset()
    if(config['text']):
        config['modelParams']['vocabSize'] = curDataset.vocabSize +1 
    #wt = ModelTraining(workerId,config,curDataset.trainData,curDataset.testData)
    trainer = getModelTrainer(config)
    trainer.setLogger(logger)
    trainer.createDataLoaders(curDataset.trainData, curDataset.testData)
    trainer.trainNEpochs(config['numEpochs'],validate=True)
    trainer.validateModel()
    
     
    
    
