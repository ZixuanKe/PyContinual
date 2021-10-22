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

def get(logger=None,args=None):

    data={}
    taskcla=[]
    # size=[3,32,32] see arg.image_size and arg.image_channel

    f_name = 'celeba_'+str(args.ntasks)
    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()
    print('random_sep: ',random_sep)

    data_celeba, taskcla_celeba = read_celeba(args=args,logger=logger)
    all_celeba = [data_celeba[x]['name'] for x in range(args.ntasks)]

    for t in range(args.ntasks):
        celeba_id = all_celeba.index(random_sep[t])
        data[t] = data_celeba[celeba_id]
        taskcla.append((t,data_celeba[celeba_id]['ncla']))

    print('taskcla: ',taskcla)
    return data,taskcla

def read_celeba(pc_valid=0.10,args=0,logger=0):
    data={}
    taskcla=[]
    # size=[3, 32, 32] see arg.image_size and arg.image_channel

    data_type = args.data_size

    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]

    # celeba
    dat={}
    #TODO:  one should save the data down since it takes a very long time



    if args.unseen and args.ntasks_unseen:
        file_name = './dat/celeba/'+data_type+'_binary_celeba_unseen/'+str(args.ntasks_unseen)+'/'
    else:
        file_name = './dat/celeba/'+data_type+'_binary_celeba/'+str(args.ntasks)+'/'


    if not os.path.isdir(file_name):
        os.makedirs(file_name)

        train_dataset = CELEBATrain(root_dir='./dat/celeba/'+data_type+'/iid/train/',img_dir='./dat/celeba/data/raw/img_align_celeba/',transform=transforms.Compose([transforms.Resize(size=(args.image_size,args.image_size)),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train'] = train_dataset

        test_dataset = CELEBATest(root_dir='./dat/celeba/'+data_type+'/iid/test/',img_dir='./dat/celeba/data/raw/img_align_celeba/',transform=transforms.Compose([transforms.Resize(size=(args.image_size,args.image_size)),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test'] = test_dataset

        users = [x[0] for x in set([user for user,image,target in torch.utils.data.DataLoader(dat['train'],batch_size=1)])] # not shuffle
        users.sort()

        if args.unseen:
            users = users[args.ntasks:args.ntasks+args.ntasks_unseen] # the first ntasks are seen, no overalpping
        else:
            users = users[:args.ntasks]
        print('users: ',users)
        print('users length: ',len(users))

        # # totally 10 tasks, each tasks 2 classes (whether smiling)
        #
        for task_id,user in enumerate(users):
            data[task_id]={}
            data[task_id]['name'] = 'celeba-'+str(task_id)
            data[task_id]['ncla'] = 2


        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1) # not shuffle

            for task_id,user in enumerate(users):
                data[task_id][s]={'x': [],'y': []}

            for user,image,target in loader:
                if user[0] not in users: continue # we dont want too may tasks
                label=target.numpy()[0]
                data[users.index(user[0])][s]['x'].append(image)
                data[users.index(user[0])][s]['y'].append(label)

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
        data[n]['name']='celeba-'+str(n)

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



# customize dataset class

class CELEBATrain(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir,img_dir, transform=None):
        self.transform = transform
        self.size=[218, 178, 3]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    for type, data in value.items():
                        if type == 'x':
                            for img in data:
                                img_name = img_dir + img
                                im = Image.open(img_name)
                                np_im = np.array(im)
                                self.x.append(torch.from_numpy(np_im))
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        self.x=torch.cat(self.x,0).view(-1,self.size[0],self.size[1],self.size[2])
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






class CELEBATest(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self, root_dir,img_dir, transform=None):
        self.transform = transform
        self.size=[218, 178, 3]

        self.x = []
        self.y = []
        self.user = []
        for file in os.listdir(root_dir):
            with open(root_dir+file) as json_file:
                data = json.load(json_file) # read file and do whatever we need to do.
                for key, value in data['user_data'].items():
                    for type, data in value.items():
                        if type == 'x':
                            for img in data:
                                img_name = img_dir + img
                                im = Image.open(img_name)
                                np_im = np.array(im)
                                self.x.append(torch.from_numpy(np_im))
                        elif type == 'y':
                            self.y.append(data)

                    for _ in range(len(data)):
                        self.user.append(key)

        self.x=torch.cat(self.x,0).view(-1,self.size[0],self.size[1],self.size[2])
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