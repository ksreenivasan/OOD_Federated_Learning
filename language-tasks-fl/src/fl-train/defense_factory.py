
from defense import *

def getDefender(conf):
    
    if conf["defenseTechnique"] == "noDefense":
        defender = None
    elif conf["defenseTechnique"] == "normClipping":
        defender = WeightDiffClippingDefense(norm_bound=conf['normBound'])
        
    elif conf["defenseTechnique"] == "weakDp":
        defender = WeakDPDefense(norm_bound=conf['normBound'])
        
    elif conf["defenseTechnique"] == "krum":
        defender = Krum(mode='krum', num_workers=conf['numActiveUsersPerRound'], num_adv=conf['numAdversaries'])
        
    elif conf["defenseTechnique"] == "multiKrum":
        defender = Krum(mode='multi-krum', num_workers=conf['numActiveUsersPerRound'], num_adv=conf['numAdversaries'])
        
    elif conf["defenseTechnique"] == "rfa":
        defender = RFA()
        
    else:
        NotImplementedError("Unsupported defense method !")
    return defender