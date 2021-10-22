import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision import datasets,transforms
import json
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from PIL import Image
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
sys.path.append(os.path.abspath('./dataloaders'))
from contrastive_dataset import InstanceSample

#TODO: please refer to celeba dataloader

def get(logger=None,args=None):
    # size=[1,28,28] see arg.image_size and arg.image_channel

    data = {}
    taskcla = []

    f_name = 'femnist_'+str(args.ntasks)
    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    data_femnist, taskcla_femnist = read_femnist(logger=logger,args=args)
    all_femnist = [data_femnist[x]['name'] for x in range(args.ntasks)]


    for task_id in range(args.ntasks):
        femnist_id = all_femnist.index(random_sep[task_id])
        data[task_id] = data_femnist[femnist_id]
        taskcla.append((task_id,data_femnist[femnist_id]['ncla']))

    print('taskcla: ',taskcla)
    return data,taskcla

def read_femnist(pc_valid=0.10,args=0,logger=0):

    print('Read FEMNIST')
    data={}
    taskcla=[]
    # size=[1,28,28]

    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}

    data_type = args.data_size

    #TODO:  one should save the data down since it takes a very long time

    if args.unseen and args.ntasks_unseen:
        file_name = './dat/femnist/'+data_type+'_binary_unseen/'+str(args.ntasks_unseen)+'/'
    else:
        file_name = './dat/femnist/'+data_type+'_binary/'+str(args.ntasks)+'/'


    if not os.path.isdir(file_name):
        os.makedirs(file_name)

        if args.unseen and args.ntasks_unseen==10:
            print('unseen')
            train_dataset = FEMMNISTTrain(root_dir='./dat/femnist/'+data_type+'/iid/train10_unseen/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]),args=args)
            dat['train'] = train_dataset

            test_dataset = FEMMNISTTest(root_dir='./dat/femnist/'+data_type+'/iid/test10_unseen/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]),args=args)
            dat['test'] = test_dataset



        elif args.ntasks==10:
            train_dataset = FEMMNISTTrain(root_dir='./dat/femnist/'+data_type+'/iid/train10/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]),args=args)
            dat['train'] = train_dataset

            test_dataset = FEMMNISTTest(root_dir='./dat/femnist/'+data_type+'/iid/test10/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]),args=args)
            dat['test'] = test_dataset


        else:
            train_dataset = FEMMNISTTrain(root_dir='./dat/femnist/'+data_type+'/iid/train20/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]),args=args)
            dat['train'] = train_dataset

            test_dataset = FEMMNISTTest(root_dir='./dat/femnist/'+data_type+'/iid/test20/',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]),args=args)
            dat['test'] = test_dataset


        users = [x[0] for x in set([user for user,image,target in torch.utils.data.DataLoader(dat['train'],batch_size=1)])]# not shuffle
        users.sort()
        print('users: ',users)
        print('users length: ',len(users))
        # # totally 47 classes, each tasks 5 classes
        #
        for task_id,user in enumerate(users):
            data[task_id]={}
            data[task_id]['name'] = 'femnist-'+str(task_id)
            data[task_id]['ncla'] = args.nclasses

        for s in ['train','test']:
            print('s: ',s)
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1)# not shuffle

            for task_id,user in enumerate(users):
                data[task_id][s]={'x': [],'y': []}

            count_label = []
            for user,image,target in loader:
                label=target.numpy()[0]

                data[users.index(user[0])][s]['x'].append(image)
                data[users.index(user[0])][s]['y'].append(label)
                count_label.append(label)

            print('count: ',Counter(count_label))

            # print('testing_c: ',testing_c)
        print('training len: ',sum([len(value['train']['x']) for key, value in data.items()]))
        print('testing len: ',sum([len(value['test']['x']) for key, value in data.items()]))

        # # "Unify" and save
        for n in range(args.ntasks):
            for s in ['train','test']:
                torch.save(data[n][s]['x'], os.path.join(os.path.expanduser(file_name),'data'+str(n)+s+'x.bin'))
                torch.save(data[n][s]['y'], os.path.join(os.path.expanduser(file_name),'data'+str(n)+s+'y.bin'))


    # # "Unify" and save
    for n in range(args.ntasks):
        data[n] = dict.fromkeys(['name','ncla','train','test'])
        for s in ['train','test']:
            data[n][s]={'x':[],'y':[]}
            data[n][s]['x']=torch.load(os.path.join(os.path.expanduser(file_name),'data'+str(n)+s+'x.bin'))
            data[n][s]['y']=torch.load(os.path.join(os.path.expanduser(file_name),'data'+str(n)+s+'y.bin'))
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,args.image_channel,args.image_size,args.image_size)
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)
        data[n]['ncla']=len(np.unique(data[n]['train']['y'].numpy()))
        data[n]['name']='femnist-'+str(n)


    # Validation
    for t in data.keys():
        r=np.arange(data[t]['train']['x'].size(0))
        r=np.array(shuffle(r,random_state=args.data_seed),dtype=int)
        # print('len(r): ',len(r))
        nvalid=int(pc_valid*len(r)) #real validataion set
        ivalid=torch.LongTensor(r[:nvalid])
        itrain=torch.LongTensor(r[nvalid:])
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
        data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
        data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
        data[t]['train']['y']=data[t]['train']['y'][itrain].clone()
        data[t]['num_train_steps'] = 0



        all_tasks = torch.tensor([t for f in data[t]['train']['y']], dtype=torch.long)


        if args.train_data_size > 0:
            # random.Random(args.data_seed).shuffle(train_data)  #unlike dsc, not neccessary to shuffle
            data[t]['train']['x'] = data[t]['train']['x'][:args.train_data_size]
            data[t]['train']['y'] = data[t]['train']['y'][:args.train_data_size]
            all_tasks = all_tasks[:args.train_data_size]
            #checked whether all tasks have samples for all set of labels. #62
            print(" data[t]['train']['y']: ", len(set(data[t]['train']['y'].cpu().numpy())))

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

# customize dataset class

class FEMMNISTTrain(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir, transform=None, args=None):
        self.transform = transform
        # self.size=[1,28,28]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    for type, data in value.items():
                        if type == 'x':
                            self.x.append(torch.from_numpy(np.array(data)))
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        #number of class
        print(len(set([b for a in self.y for b in a])))
        #number of class

        self.x=torch.cat(self.x,0).view(-1,args.image_size,args.image_size)
        self.y=torch.LongTensor(np.array([d for f in self.y for d in f],dtype=int)).view(-1).numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        user = self.user[idx]
        x = self.x[idx]
        y = self.y[idx]

        x = x.data.numpy()
        x = Image.fromarray(x)
        # x = Image.fromarray((x * 255).astype(np.uint8))

        if self.transform:
            x = self.transform(x)
        return user,x,y






class FEMMNISTTest(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir, transform=None,args=None):
        self.transform = transform
        # self.size=[1,28,28]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    for type, data in value.items():
                        if type == 'x':
                            self.x.append(torch.from_numpy(np.array(data)))
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        self.x=torch.cat(self.x,0).view(-1,args.image_size,args.image_size)
        self.y=torch.LongTensor(np.array([d for f in self.y for d in f],dtype=int)).view(-1).numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        user = self.user[idx]
        x = self.x[idx]
        y = self.y[idx]

        x = x.data.numpy()
        x = Image.fromarray(x)
        # x = Image.fromarray((x * 255).astype(np.uint8))

        if self.transform:
            x = self.transform(x)
        return user,x,y

