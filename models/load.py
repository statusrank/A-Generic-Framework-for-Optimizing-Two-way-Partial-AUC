import os
import torch

def load_pretrained_model(model, state_dict):
    model_state = model.state_dict()
    model_params = len(model_state.keys())
    checkpoint_params = len(state_dict.keys())
    # logger.info('this model has {} params; this checkpoint has {} params'.format(model_params,checkpoint_params))
    print('this model has {} params; this checkpoint has {} params'.format(model_params,checkpoint_params))
    if model_params > checkpoint_params:
        for i,param in model_state.items():
            if i not in state_dict.keys():
                # logger.info('this param of the model dont in the checkpoint: {} ,required grad: {}'.format(i,str(param.requires_grad)))
                print('this param of the model dont in the checkpoint: {} ,required grad: {}'.format(i,str(param.requires_grad)))
    num = 0
    total = 0
    for k,v in state_dict.items():
        total += 1
        if k in model_state.keys():
            if not isinstance(v,bool):
                if (v.size() != model_state[k].size()):
                    # logger.info('this param {} of the checkpoint dont match the model in size: '.format(k) + str(v.size()) + ' ' + str(model_state[k].size()))
                    print('this param {} of the checkpoint dont match the model in size: '.format(k) + str(v.size()) + ' ' + str(model_state[k].size()))
                    continue
            model_state[k] = v
            num += 1
        else:
            # logger.info('this param of the checkpoint dont in the model: {}'.format(k))
            print('this param of the checkpoint dont in the model: {}'.format(k))
    model.load_state_dict(model_state, strict=False)
    # logger.info('success for loading pretrained model params {}/{}!'.format(str(num), str(total)))
    print('success for loading pretrained model params {}/{}!'.format(str(num), str(total)))
