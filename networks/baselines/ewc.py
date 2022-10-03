


from tqdm.auto import tqdm
import torch
import torch.distributed as dist
import os
from utils import utils
import math

def compute(train_dataloader_prune,model,self_fisher,accelerator,args):
    fisher_path = os.path.join(args.output_dir, 'fisher')

    if args.ft_task > 0:
        fisher_old = {}
        for n, _ in model.named_parameters():
            fisher_old[n] = self_fisher[n].clone()


    # Init
    progress_bar = tqdm(range(len(train_dataloader_prune)), disable=not accelerator.is_local_main_process)


    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()

    for step, inputs in enumerate(train_dataloader_prune):
        model.zero_grad()
        # if we merge everything beforehead, we will see NAN
        sbatch = inputs['input_ids'].size(0)


        outputs = model(inputs=inputs,self_fisher=self_fisher)

        loss = outputs.loss  # loss 1

        loss = loss / args.gradient_accumulation_steps


        # add model needs to be careful! make sure it is in parameters and please double check its gradient
        accelerator.backward(loss)  # sync
        progress_bar.update(1)
        progress_bar.set_description('EWC Fisher Compute Iter (loss=%5.3f)' % loss.item())  # show the loss, mean while
        # Get model
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)

    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/len(train_dataloader_prune)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)

    self_fisher = fisher

    if args.ft_task > 0:
        # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
        for n, _ in model.named_parameters():
            self_fisher[n] = (self_fisher[n] + fisher_old[n] * args.ft_task) / (args.ft_task + 1)  # Checked: it is better than the other option
            # self.fisher[n]=0.5*(self.fisher[n]+fisher_old[n])

    accelerator.wait_for_everyone()

    for k,v in self_fisher.items():
        self_fisher[k] = utils.gather_mean(self_fisher[k])

    if accelerator.is_main_process:
        torch.save(self_fisher, fisher_path)

    return fisher

#TODO: incorrect === error

def loss_compute(my_model,self_fisher):
    loss_reg = 0
    if my_model.args.ft_task > 0:

        for (name, param), (_, param_old) in zip(my_model.model.named_parameters(),
                                                 my_model.teacher.named_parameters()):  # has to bealigned

            if 'classifier' not in name:  # no need to include the classifier
                cur_loss_reg = torch.sum(
                    self_fisher['module.model.' + name] * (param_old.cuda() - param.cuda()).pow(2)) / 2
                if torch.isnan(cur_loss_reg) or torch.isinf(cur_loss_reg):
                    continue
                else:
                    loss_reg += cur_loss_reg

        loss = my_model.args.lamb * loss_reg # use above for robustness

    return loss
