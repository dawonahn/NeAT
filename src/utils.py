import os
import torch

        
def save_checkpoints(model, config):
    '''Save a trained model.'''
    data_config = os.path.join(config.dataset)
    model_config = f"{config.nn}_{config.rank}"
    out_path = os.path.join(config.rpath, "output", data_config, model_config)
    model_path = os.path.join(out_path, f'{config.wnb_name}.pt')

    os.makedirs(out_path, exist_ok=True)
    torch.save(dict(model_state=model.state_dict()), model_path)

    return model_path