import os
import sys
sys.path.append('./src/')
sys.path.append('./src/fl-train/')
sys.path.append('./src/training/')
sys.path.append('./src/dataset/*')
import argparse
from fl_train import FLTrainer
from globalUtils import *

def getConfParamVal(key,conf):
    keys = key.split('.')
    y = conf
    for k in keys:
        y = y[k]
    return y

if __name__ == "__main__":

  
    parser = argparse.ArgumentParser(description='Process args')
    parser.add_argument('--config', type=str,
                        help='The conf file to be used for the training')

    args = parser.parse_args()
    
    seed(42)
    confFilePath = args.config
    
    conf = loadConfig(confFilePath)

    od = conf['outputDir']
    if(od.startswith('$')):
        od = od[1:]
        x = od.split('/')
        name = '_'.join( [ '{}_{}'.format(k, getConfParamVal(k,conf)) for k in x[-1].split('_') ] )
        od= '/'.join(x[:-1])+'/'+name
        conf['outputDir'] = od
    
    if not os.path.exists(od):
        os.makedirs(od)
    print('Will be saving output and logs in directory {}'.format(od))
    
    stdoutFlag = True
    logger = getLogger("{}/fl.log".format(od), stdoutFlag, logging.INFO)
    print('Log File: {}/fl.log'.format(od))
    print(conf['ckptEpochs'])
    
    flTrainer = FLTrainer(conf,logger)
    flTrainer.trainNEpochs()
    
