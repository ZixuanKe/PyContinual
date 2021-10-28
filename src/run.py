import sys,os,argparse,time
import numpy as np
import torch
from config import set_args
import utils
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import RandomSampler,SubsetRandomSampler
import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
import os
import random
from preparation import *

import torch.nn as nn
import pickle
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# Arguments
#
#

# ########################################################################################################################

# Args -- Experiment
import import_classification as import_modules

# ########################################################################################################################
# ----------------------------------------------------------------------
# Load Data.
# ----------------------------------------------------------------------
print('Load data...')
if args.experiment=='w2v' or args.experiment=='w2v_as':
    data,taskcla,vocab_size,embeddings=import_modules.dataloader.get(logger=logger,args=args)
else:
    data,taskcla=import_modules.dataloader.get(logger=logger,args=args)

print('\nTask info =',taskcla)
#
# Inits
print('Inits...')


# ----------------------------------------------------------------------
# Apply approach and network.
# ----------------------------------------------------------------------

if 'owm' in args.approach:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'bert' in args.approach or 'cnn' in args.approach or 'mlp' in args.approach:
        net = import_modules.network.Net(taskcla,args=args)
    elif 'w2v' in args.approach:
        net=import_modules.network.Net(taskcla,embeddings,args=args)


elif 'ucl' in args.approach:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'bert' in args.approach or 'cnn' in args.approach or 'mlp' in args.approach:
        net = import_modules.network.Net(taskcla,args=args)
        net_old = import_modules.network.Net(taskcla,args=args)

    elif 'w2v' in args.approach:
        net = import_modules.network.Net(taskcla, embeddings, args.ratio,args=args)
        net_old = import_modules.network.Net(taskcla, embeddings, args.ratio,args=args)
else:
    if args.aux_net: #this is for 2 network setting
        print('net')
        args.is_aux = False
        net=import_modules.network.Net(taskcla,args=args)
        print('aux net')
        args.is_aux = True
        aux_net=import_modules.aux_network.Net(taskcla,args=args)


    if 'w2v' in args.approach:
        net=import_modules.network.Net(taskcla,embeddings,args=args)
    else:
        net=import_modules.network.Net(taskcla,args=args)
    logger.info('count: '+str(torch.cuda.device_count()))


if 'net' in locals(): net = net.to(device)
if 'aux_net' in locals(): aux_net = aux_net.to(device)
if 'net_old' in locals(): net_old = net_old.to(device)



if args.aux_net:
    appr=import_modules.approach.Appr(net,aux_net,logger=logger,taskcla=taskcla,args=args)
else:
    appr=import_modules.approach.Appr(net,logger=logger,taskcla=taskcla,args=args)

if not args.eval_each_step:
    if args.aux_net:
        resume_checkpoint(appr,net,aux_net)
    else:
        resume_checkpoint(appr,net,None)


if args.multi_gpu and args.distributed:

    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=n_gpu)
    net = net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)
    #TODO: distribuited may be hang and stuck here

elif args.multi_gpu:
    logger.info('multi_gpu')
    net = torch.nn.DataParallel(net)
    net = net.to(device)
    if args.aux_net:
        aux_net = torch.nn.DataParallel(aux_net)
        aux_net = aux_net.to(device)

if args.print_report:
    utils.print_model_report(net)
# print(appr.criterion)
# utils.print_optimizer_config(appr.optimizer)
# print('-'*100)



# ----------------------------------------------------------------------
# Start Training.
# ----------------------------------------------------------------------

for t,ncla in taskcla:


    if args.eval_each_step:
        args.resume_from_aux_file = base_resume_from_aux_file + 'steps'+str(t)
        args.resume_from_file = base_resume_from_file + 'steps'+str(t)
        resume_checkpoint(appr)

    # print('*'*100)
    # print('Task {:2d} ({:s})'.format(t,data[t]['name']))
    # print('*'*100)
    #
    logger.info('*'*100)
    logger.info('Task {:2d} ({:s})'.format(t,data[t]['name']))
    logger.info('*'*100)

    # if t>1: exit()

    if 'mtl' in args.approach:
        # Get data. We do not put it to GPU
        if t==0:
            train=data[t]['train']
            valid=data[t]['valid']
            num_train_steps=data[t]['num_train_steps']

        else:
            train = ConcatDataset([train,data[t]['train']])
            valid = ConcatDataset([valid,data[t]['valid']])
            num_train_steps+=data[t]['num_train_steps']
        task=t

        if t < len(taskcla)-1: continue #only want the last one

    else:
        # Get data
        train=data[t]['train']
        valid=data[t]['valid']
        num_train_steps=data[t]['num_train_steps']
        task=t


    if  args.task == 'asc': #special setting
        if 'XuSemEval' in data[t]['name']:
            args.num_train_epochs=args.xusemeval_num_train_epochs #10
        else:
            args.num_train_epochs=args.bingdomains_num_train_epochs #30
            num_train_steps*=args.bingdomains_num_train_epochs_multiplier # every task got refresh, *3

    if args.multi_gpu and args.distributed:
        valid_sampler = DistributedSampler(valid) #TODO: DitributedSequentailSampler
        valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size)
    else:
        valid_sampler = SequentialSampler(valid)
        valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size,pin_memory=True)

    if args.resume_model and t < args.resume_from_task: continue #resume. dont forget to copy the forward results


    if args.multi_gpu and args.distributed:
        train_sampler = DistributedSampler(train)
        train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size)
    else:
        train_sampler = RandomSampler(train)
        train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size,pin_memory=True)

    logger.info('Start Training and Set the clock')
    tstart=time.time()


    if not args.eval_only:
        if args.task in extraction_tasks:
            label_list = data[t]['label_list']
            appr.train(task,train_dataloader,valid_dataloader,num_train_steps,train,valid,label_list)
        else:
            # Train
            print('train')
            appr.train(task,train_dataloader,valid_dataloader,num_train_steps,train,valid)

    print('-'*100)

    if args.exit_after_first_task: #sometimes we want to fast debug or estimate the excution time
        #TODO: consider save to file and print them out
        logger.info('[Elapsed time per epochs = {:.1f} s]'.format((time.time()-tstart)))
        logger.info('[Elapsed time per epochs = {:.1f} min]'.format((time.time()-tstart)/(60)))
        logger.info('[Elapsed time per epochs = {:.1f} h]'.format((time.time()-tstart)/(60*60)))
        if 'kim' not in args.approach and 'mlp' not in args.approach and 'cnn' not in args.approach:
            if 'asc' in args.task: pre_define_num_epochs = 30 #non-semeval estimation
            elif 'dsc' in args.task: pre_define_num_epochs = 20
            elif 'newsgroup' in args.task: pre_define_num_epochs = 10
            logger.info('[Elapsed time per tasks = {:.1f} s]'.format((time.time()-tstart)*pre_define_num_epochs))
            logger.info('[Elapsed time per tasks = {:.1f} min]'.format(((time.time()-tstart)/(60))*pre_define_num_epochs))
            logger.info('[Elapsed time per tasks = {:.1f} h]'.format(((time.time()-tstart)/(60*60))*pre_define_num_epochs))
        else:
            if 'asc' in args.task: additional = 3 #different size for asc tasks
            else: additional = 1
            logger.info('[Elapsed time per tasks = {:.1f} s]'.format((time.time()-tstart)*50*additional)) # estimation for early stopping
            logger.info('[Elapsed time per tasks = {:.1f} min]'.format(((time.time()-tstart)/(60))*50*additional))
            logger.info('[Elapsed time per tasks = {:.1f} h]'.format(((time.time()-tstart)/(60*60))*50*additional))
        exit()

    if args.save_each_step:
        args.model_path = base_model_path + 'steps'+str(t)
        args.aux_model_path = base_aux_model_path + 'steps'+str(t)

    if args.save_model:
        print('save model ')

        torch.save({
                    'model_state_dict': appr.model.state_dict(),
                    }, args.model_path)
        #for GEM
        if hasattr(appr, 'buffer'): torch.save(appr.buffer,args.model_path+'_buffer') # not in state_dict
        if hasattr(appr, 'grad_dims'): torch.save(appr.grad_dims,args.model_path+'_grad_dims') # not in state_dict
        if hasattr(appr, 'grads_cs'): torch.save(appr.grads_cs,args.model_path+'_grads_cs') # not in state_dict
        if hasattr(appr, 'grads_da'): torch.save(appr.grads_da,args.model_path+'_grads_da') # not in state_dict
        if hasattr(appr, 'history_mask_pre'): torch.save(appr.history_mask_pre,args.model_path+'_history_mask_pre') # not in state_dict
        if hasattr(appr, 'similarities'): torch.save(appr.similarities,args.model_path+'_similarities') # not in state_dict
        if hasattr(appr, 'check_federated'): torch.save(appr.check_federated,args.model_path+'_check_federated') # not in state_dict



        if args.aux_net:
            torch.save({
                        'model_state_dict': appr.aux_model.state_dict(),
                        }, args.aux_model_path)
            if hasattr(appr, 'mask_pre'): torch.save(appr.mask_pre,args.aux_model_path+'_mask_pre') # not in state_dict
            if hasattr(appr, 'mask_back'): torch.save(appr.mask_back,args.aux_model_path+'_mask_back')
        else:
            if hasattr(appr, 'mask_pre'): torch.save(appr.mask_pre,args.model_path+'_mask_pre') # not in state_dict
            if hasattr(appr, 'mask_back'): torch.save(appr.mask_back,args.model_path+'_mask_back')

    # ----------------------------------------------------------------------
    # Start Testing.
    # ----------------------------------------------------------------------

    if args.unseen and args.eval_each_step: #we want to test every one for unseen
        test_set = args.ntasks
    else:
        test_set = t+1
    for u in range(test_set):

        test=data[u]['test']

        if args.multi_gpu and args.distributed:
            test_sampler = DistributedSampler(test)
            test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)
        else:
            test_sampler = SequentialSampler(test)
            test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)

        if args.task in classification_tasks: #classification task

            if 'kan' in args.approach:
                test_loss,test_acc,test_f1_macro=appr.eval(u,test_dataloader,test,which_type='mcl',trained_task=t)
                logger.info('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
            elif 'cat' in args.approach:
                valid=data[u]['valid']
                valid_sampler = SequentialSampler(valid)
                valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size,pin_memory=True)
                test_loss,test_acc,test_f1_macro=appr.eval(u,test_dataloader,valid_dataloader,trained_task=t,phase='mcl')
                logger.info('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))

            else:
                test_loss,test_acc,test_f1_macro=appr.eval(u,test_dataloader,test,trained_task=t)
                logger.info('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))


            # elif args.eval_only: pass
            # else:
            acc[t,u]=test_acc
            lss[t,u]=test_loss
            f1_macro[t,u]=test_f1_macro

            # Save
            print('Save at '+args.output)
            np.savetxt(args.output + 'progressive.acc',acc,'%.4f',delimiter='\t')
            np.savetxt(args.output + 'progressive.f1_macro',f1_macro,'%.4f',delimiter='\t')

            # Done
            print('*'*100)
            print('Accuracies =')
            for i in range(acc.shape[0]):
                print('\t',end='')
                for j in range(acc.shape[1]):
                    print('{:5.1f}% '.format(100*acc[i,j]),end='')
                print()
            print('*'*100)
            print('Done!')

            print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))


            with open(performance_output,'w') as file, open(f1_macro_output,'w') as f1_file:
                if 'ncl' in args.approach  or 'mtl' in args.approach:
                    for j in range(acc.shape[1]):
                        file.writelines(str(acc[-1][j]) + '\n')
                        f1_file.writelines(str(f1_macro[-1][j]) + '\n')

                elif 'one' in args.approach:
                    for j in range(acc.shape[1]):
                        file.writelines(str(acc[j][j]) + '\n')
                        f1_file.writelines(str(f1_macro[j][j]) + '\n')


            with open(performance_output_forward,'w') as file, open(f1_macro_output_forward,'w') as f1_file:
                if 'ncl' in args.approach  or 'mtl' in args.approach:
                    for j in range(acc.shape[1]):
                        file.writelines(str(acc[j][j]) + '\n')
                        f1_file.writelines(str(f1_macro[j][j]) + '\n')


########################################################################################################################

