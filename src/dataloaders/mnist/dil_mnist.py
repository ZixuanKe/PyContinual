import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision import datasets,transforms
import json
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
sys.path.append(os.path.abspath('./dataloaders'))
from contrastive_dataset import InstanceSample
import random
from sklearn.model_selection import train_test_split
def get(logger=None,args=None):

    data={}
    taskcla=[]
    # size=[3,32,32] see arg.image_size and arg.image_channel

    f_name = 'mnist_'+str(args.ntasks)
    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()
    print('random_sep: ',random_sep)

    data_mnist, taskcla_mnist = read_mnist(args=args,logger=logger)
    all_mnist = [data_mnist[x]['name'] for x in range(args.ntasks)]

    for t in range(args.ntasks):
        mnist_id = all_mnist.index(random_sep[t])
        data[t] = data_mnist[mnist_id]
        taskcla.append((t,data_mnist[mnist_id]['ncla']))

    print('taskcla: ',taskcla)
    return data,taskcla

def read_mnist(pc_valid=0.10,args=0,logger=0):
    data={}
    taskcla=[]
    # size=[3, 32, 32] see arg.image_size and arg.image_channel

    mean=(0.1307,)
    std=(0.3081,)

    # mnist
    dat={}
    #TODO:  one should save the data down since it takes a very long time

    if not os.path.isdir('./dat/mnist/'+str(args.ntasks)+'/'):
        os.makedirs('./dat/mnist/'+str(args.ntasks)+'/')


        candidate_train=datasets.MNIST('./dat/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        candidate_test=datasets.MNIST('./dat/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    
        loader=torch.utils.data.DataLoader(candidate_train,batch_size=1) # not shuffle
        candidate_x_train = []
        candidate_y_train = []
        for image,target in loader:
            candidate_x_train.append(image)
            candidate_y_train.append(target)

        loader=torch.utils.data.DataLoader(candidate_test,batch_size=1) # not shuffle
        candidate_x_test = []
        candidate_y_test = []
        for image,target in loader:
            candidate_x_test.append(image)
            candidate_y_test.append(target)


        for task_id in range(args.ntasks):
            data[task_id]={}
            data[task_id]['name'] = 'mnist-'+str(task_id)
            data[task_id]['ncla'] = 10



            if 1/(args.ntasks-task_id) == 1:
                current_x_train = candidate_x_train
                current_y_train = candidate_y_train
                current_x_test = candidate_x_test
                current_y_test = candidate_y_test

            else:
                candidate_x_train, current_x_train, candidate_y_train, current_y_train = train_test_split(candidate_x_train, candidate_y_train, test_size=1/(args.ntasks-task_id), stratify=candidate_y_train)
                candidate_x_test, current_x_test, candidate_y_test, current_y_test = train_test_split(candidate_x_test, candidate_y_test, test_size=1/(args.ntasks-task_id), stratify=candidate_y_test)


            data[task_id]['train']={'x': [],'y': []}
            data[task_id]['train']['x'] += current_x_train
            data[task_id]['train']['y']+= current_y_train
            
            data[task_id]['test']={'x': [],'y': []}
            data[task_id]['test']['x'] +=current_x_test
            data[task_id]['test']['y'] +=current_y_test


        # # "Unify" and save
        for n in range(args.ntasks):
            for s in ['train','test']:
                torch.save(data[n][s]['x'], os.path.join(os.path.expanduser('./dat/mnist/'+str(args.ntasks)),'data'+str(n)+s+'x.bin'))
                torch.save(data[n][s]['y'], os.path.join(os.path.expanduser('./dat/mnist/'+str(args.ntasks)),'data'+str(n)+s+'y.bin'))


    # # "Unify" and save
    for n in range(args.ntasks):
        data[n] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[n][s]={'x':[],'y':[]}
            data[n][s]['x']=torch.load(os.path.join(os.path.expanduser('./dat/mnist/'+str(args.ntasks)),'data'+str(n)+s+'x.bin'))
            data[n][s]['y']=torch.load(os.path.join(os.path.expanduser('./dat/mnist/'+str(args.ntasks)),'data'+str(n)+s+'y.bin'))
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,args.image_channel,args.image_size,args.image_size)
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
        data[n]['ncla']=len(np.unique(data[n]['train']['y'].numpy()))
        data[n]['name']='mnist-'+str(n)

    # Real Validation
    for t in data.keys():

        logger.info(data[t]['name'])

        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=args.data_seed),dtype=int)  # seed set
        nvalid=int(pc_valid*len(r))
        ivalid=torch.LongTensor(r[:nvalid])

        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()
        data[t]['num_train_steps'] = 0

        all_tasks = torch.tensor([t for f in data[t]['train']['y']], dtype=torch.long)

        #checked whether all tasks have samples for all set of labels. #62
        print(" data[t]['train']['y']: ", len(set(data[t]['train']['y'].cpu().numpy())))



        if args.train_data_size > 0:
            # random.Random(args.data_seed).shuffle(train_data)  #unlike dsc, not neccessary to shuffle
            data[t]['train']['x'] = data[t]['train']['x'][:args.train_data_size]
            data[t]['train']['y'] = data[t]['train']['y'][:args.train_data_size]
            all_tasks = all_tasks[:args.train_data_size]

        if args.distill_loss:
            train_data = InstanceSample(
                data[t]['train']['x'],
                data[t]['train']['y']
                )
        elif args.mtl:
            train_data = TensorDataset(
                data[t]['train']['x'],
                data[t]['train']['y'],
                all_tasks
                )
        else:
            train_data = TensorDataset(
                data[t]['train']['x'],
                data[t]['train']['y']
                )


        data[t]['train']=train_data


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))

        all_tasks = torch.tensor([t for f in data[t]['valid']['y']], dtype=torch.long)
        if args.dev_data_size > 0:
            # random.Random(args.data_seed).shuffle(eval_data) #unlike dsc, not neccessary to shuffle
            data[t]['valid']['x'] = data[t]['valid']['x'][:args.dev_data_size]
            data[t]['valid']['y'] = data[t]['valid']['y'][:args.dev_data_size]
            all_tasks = all_tasks[:args.dev_data_size]

        if args.distill_loss:
            eval_data = InstanceSample(
                data[t]['valid']['x'],
                data[t]['valid']['y']
                )
        elif args.mtl:
            eval_data = TensorDataset(
                data[t]['valid']['x'],
                data[t]['valid']['y'],
                all_tasks
                )
        else:
            eval_data = TensorDataset(
                data[t]['valid']['x'],
                data[t]['valid']['y']
                )

        data[t]['valid']=eval_data

        logger.info("***** Running Dev *****")
        logger.info("  Num examples = %d", len(eval_data))

        all_tasks = torch.tensor([t for f in data[t]['test']['y']], dtype=torch.long)
        if args.test_data_size > 0:
            # random.Random(args.data_seed).shuffle(test_data)  #unlike dsc, not neccessary to shuffle
            data[t]['test']['x'] = data[t]['test']['x'][:args.test_data_size]
            data[t]['test']['y'] = data[t]['test']['y'][:args.test_data_size]
            all_tasks = all_tasks[:args.test_data_size]

        if args.distill_loss:
            test_data = InstanceSample(
                data[t]['test']['x'],
                data[t]['test']['y']
                )
        elif args.mtl:
            test_data = TensorDataset(
                data[t]['test']['x'],
                data[t]['test']['y'],
                all_tasks
                )
        else:
            test_data = TensorDataset(
                data[t]['test']['x'],
                data[t]['test']['y']
                )

        data[t]['test']=test_data

        logger.info("***** Running testing *****")
        logger.info("  Num examples = %d", len(test_data))

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla


########################################################################################################################


