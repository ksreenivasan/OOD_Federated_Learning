from training.text_bc_training import TextBCModelTraining
from training.generic_model_training import GenericModelTraining

def getModelTrainer(conf,lr=None,isAttacker=False):
    model = conf['arch']
    if(model=='textBC'):
        return TextBCModelTraining(conf,lr,isAttacker=isAttacker)
    if(model=='lenet5'):
        return GenericModelTraining(conf,lr,isAttacker=isAttacker)
    
