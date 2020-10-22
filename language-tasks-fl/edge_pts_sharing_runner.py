import os
import sys
sys.path.append('./src/')
sys.path.append('./src/fl-train/')
sys.path.append('./src/training/')
sys.path.append('./src/dataset/*')
import argparse
from fl_train import FLTrainer
from globalUtils import *
import copy


from multiprocessing import Process

def getConfParamVal(key,conf):
    keys = key.split('.')
    y = conf
    for k in keys:
        y = y[k]
    return y

def runConf(conf):
    seed(42)
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

    stdoutFlag = False
    logger = getLogger("{}/fl.log".format(od), stdoutFlag, logging.INFO)
    print('Log File: {}/fl.log'.format(od))
    #print(conf['ckptEpochs'])

    flTrainer = FLTrainer(conf,logger)
    flTrainer.trainNEpochs()

import time


def ParallelRun(lstConf):
    lstP = []
    for conf in lstConf:
        conf = copy.deepcopy(conf) # ensure no shit happens
        p = Process(target = runConf, args=(conf,))
        p.start()
        lstP.append(p)
        
    for p in lstP:
        p.join()
        
def SeqRun(lstConf):
    for conf in lstConf:
        runConf(conf)
    
if __name__ == "__main__":

  
    parser = argparse.ArgumentParser(description='Process args')
    parser.add_argument('--config', type=str,
                        help='The conf file to be used for the training')

    args = parser.parse_args()
        
    confFilePath = args.config
    conf0 = loadConfig(confFilePath)
    
    numEdgePtsAdv: 20
    numEdgePtsGood: 0
     
    lstConf = []
    
    conf0['attackFromEpoch']=100
    conf0['numFLEpochs'] = 300
    conf0['enableCkpt'] = False
    
    op_pfx = './outputs/yorgos_backdoor_'
    #lstNumEdgePtsAdv = [10, 20, 40 ,60, 80, 100,120]
    #lstNumEdgePtsGood = [0, 20, 40, 60,80, 100,120]
    lstPairs = [(10,90),(50,50),(90,10),(20,180),(100,100),(180,20),(10,190),(40,160)]
    for p in lstPairs:
        a = p[0]
        b = p[1]

        if(a==0 and b==0):
            continue
        conf = copy.deepcopy(conf0)
        conf['numEdgePtsAdv'] = a
        conf['numEdgePtsGood'] = b
        conf['outputDir'] = '{}_numEdgeAdv_{}_numEdgeGood_{}/'.format(op_pfx,a,b)
        lstConf.append(conf)
    
    parBatch = 5   
    # Execution
    if(parBatch>1):
        for i in range(len(lstConf)):
            ParallelRun(lstConf[i*parBatch:(i+1)*parBatch])
    else:
        SeqRun(lstConf)
        
    print('Here')
          
        
    
