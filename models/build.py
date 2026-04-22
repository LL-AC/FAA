import timm
import json
import torch
from .FAA_tmed import FAA_tmed

def build_model(args):
    
    model_name = args.model.lower()
    
    if model_name == 'faatmed':
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        print(args.config_path, 'loading model config')
        model = FAA_tmed(n_classes=args.num_classes, config=config)
    else:
        raise NotImplementedError(f"{model_name} is not implemented")
    
    return model