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

    f_name = 'cifar10_'+str(args.ntasks)
    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()
    print('random_sep: ',random_sep)

    data_cifar10, taskcla_cifar10 = read_cifar10(args=args,logger=logger)
    all_cifar10 = [data_cifar10[x]['name'] for x in range(args.ntasks)]

    for t in range(args.ntasks):
        cifar10_id = all_cifar10.index(random_sep[t])
        data[t] = data_cifar10[cifar10_id]
        taskcla.append((t,data_cifar10[cifar10_id]['ncla']))

    print('taskcla: ',taskcla)
    return data,taskcla


def hetero_partition(args,K,N,y):
    net_dataidx_map = {}

    idx_batch = [[] for _ in range(args.ntasks)]
    # for each class in the dataset
    for k in range(K):
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(0.5, args.ntasks))
        ## Balance
        proportions = np.array([p*(len(idx_j)<N/args.ntasks) for p,idx_j in zip(proportions,idx_batch)])
        proportions = proportions/proportions.sum()
        proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]

        print('proportions: ',proportions) #proportion is a soted 1-D array

        visited_ele = []
        for ele_id,ele in enumerate(proportions):

            if proportions[0] == 0:
                alloc = proportions[1] // 2
                proportions[0] += alloc

            if ele in visited_ele: #some classes has 0 samples
                alloc = (visited_ele[visited_ele.index(ele)] - visited_ele[visited_ele.index(ele)-1]) // 2
                proportions[visited_ele.index(ele)]-= alloc
                visited_ele[visited_ele.index(ele)]-= alloc
            visited_ele.append(ele)

        print('proportions: ',proportions) #proportion is a soted 1-D array



        print('np.split(idx_k,proportions): ',len(np.split(idx_k,proportions)))
        idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))] #split and corresponding to different task
        print('idx_batch: ',len(idx_batch))

    for j in range(args.ntasks):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

def read_cifar10(pc_valid=0.10,args=0,logger=0):
    data={}
    taskcla=[]
    # size=[3, 32, 32] see arg.image_size and arg.image_channel

    data_type = args.data_size

    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]

    # cifar10
    dat={}

    #TODO:  one should save the data down since it takes a very long time if reload every time
    if args.hetero: path_name = './dat/cifar10/'+str(args.ntasks)+'/'+'hetero/'
    else: path_name = './dat/cifar10/'+str(args.ntasks)+'/'+'homo/'
    if not os.path.isdir(path_name):
        os.makedirs(path_name)

        candidate_train=datasets.CIFAR10('./dat/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
        candidate_test=datasets.CIFAR10('./dat/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

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

        if args.hetero:

            candidate_y = candidate_y_train + candidate_y_test
            #reallocate the trainig and testing set. since the training set are now become non-i.i.d, while we want trainig and testing are i.i.d
            candidate_x = candidate_x_train + candidate_x_test

            candidate_y = np.array(candidate_y)

            K = 10 #class
            N = candidate_y.shape[0]
            net_dataidx_map = hetero_partition(args,K,N,candidate_y)

            net_dataidx_map_train = []
            net_dataidx_map_test = []

            for task_id in range(args.ntasks):
                net_dataidx_map_train.append(net_dataidx_map[task_id][:int(len(net_dataidx_map[task_id])*0.8)])
                net_dataidx_map_test.append(net_dataidx_map[task_id][int(len(net_dataidx_map[task_id])*0.8):])


        for task_id in range(args.ntasks):
            data[task_id]={}
            data[task_id]['name'] = 'cifar10-'+str(task_id)
            data[task_id]['ncla'] = 10

            if args.hetero:
                data[task_id]['train']={'x': [],'y': []}
                data[task_id]['train']['x'] += [candidate_x[i] for i in net_dataidx_map_train[task_id]]
                data[task_id]['train']['y']+= [candidate_y[i] for i in net_dataidx_map_train[task_id]]

                data[task_id]['test']={'x': [],'y': []}
                data[task_id]['test']['x'] +=[candidate_x[i] for i in net_dataidx_map_test[task_id]]
                data[task_id]['test']['y'] +=[candidate_y[i] for i in net_dataidx_map_test[task_id]]

            else:


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
                torch.save(data[n][s]['x'], os.path.join(os.path.expanduser(path_name),'data'+str(n)+s+'x.bin'))
                torch.save(data[n][s]['y'], os.path.join(os.path.expanduser(path_name),'data'+str(n)+s+'y.bin'))


    # # "Unify" and save
    for n in range(args.ntasks):
        data[n] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[n][s]={'x':[],'y':[]}
            data[n][s]['x']=torch.load(os.path.join(os.path.expanduser(path_name),'data'+str(n)+s+'x.bin'))
            data[n][s]['y']=torch.load(os.path.join(os.path.expanduser(path_name),'data'+str(n)+s+'y.bin'))
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,args.image_channel,args.image_size,args.image_size)
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
        data[n]['ncla']=len(np.unique(data[n]['train']['y'].numpy()))
        data[n]['name']='cifar10-'+str(n)

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

        if args.mtl:
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

        if args.mtl:
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

        if args.mtl:
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

    # exit()
    return data,taskcla


########################################################################################################################


