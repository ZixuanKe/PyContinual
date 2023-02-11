


from tqdm.auto import tqdm
import torch
import torch.distributed as dist
import os
import utils
import numpy as np




def compute(model,accelerator,mask_pre,mask,args):
    mask_pre_path = os.path.join(args.output_dir, 'mask_pre') # we don't want mask_pre to be overlapping with others, imagine if you stop and rerun some
    mask_back_path = os.path.join(args.output_dir, 'mask_back')
    model_ori = accelerator.unwrap_model(model)
    config = model_ori.config


    for key, value in mask.items():
        mask[key] = torch.autograd.Variable(value.data.clone(), requires_grad=False)
    if args.ft_task == 0:
        mask_pre = mask
    else:
        for key, value in mask_pre.items():
            mask_pre[key] = torch.max(mask_pre[key], mask[key])

    # Weights mask
    mask_back = {}
    for n, p in model.named_parameters():
        vals = utils.model.get_view_for(n, p, mask_pre,config, args)
        if vals is not None:
            mask_back[n] = 1 - vals


    accelerator.wait_for_everyone()

    # n_gpu = torch.cuda.device_count()
    # print('n_gpu: ',n_gpu)
    # if n_gpu > 1:
    for k, v in mask_pre.items():
        mask_pre[k] = utils.model.gather_mean(mask_pre[k])

    for k, v in mask_back.items():
        mask_back[k] = utils.model.gather_mean(mask_back[k])

    if accelerator.is_main_process:
        torch.save(mask_pre, mask_pre_path)
        torch.save(mask_back, mask_back_path)



def loss_compute(masks,mask_pre,args):
    reg = 0
    count = 0

    if mask_pre is not None:
        # for m,mp in zip(masks,self.mask_pre):
        for key in set(masks.keys()) & set(mask_pre.keys()):
            m = masks[key]
            mp = mask_pre[key]
            aux = 1 - mp
            reg += (m * aux).sum()
            count += aux.sum()
    else:
        for m_key, m_value in masks.items():
            reg += m_value.sum()
            count += np.prod(m_value.size()).item()

    reg /= count

    loss = args.lamb * reg

    return loss