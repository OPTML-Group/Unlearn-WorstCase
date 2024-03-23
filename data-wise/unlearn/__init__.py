from .RL import RL, w_RL
from .FT import FT, FT_l1, w_FT, w_FT_l1, EU, CF, w_EU

from .retrain import retrain, w_retrain
from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint, \
    optimize_select, reverse_optimize_select

from .boundary_ex import boundary_expanding, w_boundary_expanding
from .boundary_sh import boundary_shrink, w_boundary_shrink

from .SCRUB import SCRUB, w_SCRUB


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""

    if name == "RL":
        return RL
    elif name == "w_RL":
        return w_RL
    
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "w_FT":
        return w_FT
    elif name == "w_FT_l1":
        return w_FT_l1
    
    elif name == "retrain":
        return retrain
    elif name == "w_retrain":
        return w_retrain
    
    elif name == "boundary_expanding":
        return boundary_expanding
    elif name == "boundary_shrink":
        return boundary_shrink
    elif name == "w_boundary_expanding":
        return w_boundary_expanding
    elif name == "w_boundary_shrink":
        return w_boundary_shrink
    
    elif name == "scrub":
        return SCRUB
    elif name == "w_scrub":
        return w_SCRUB
    
    elif name == "EU":
        return EU
    elif name == "CF":
        return CF
    elif name == "w_EU":
        return w_EU
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
