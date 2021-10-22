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


domains = [
     'sun',
     'pascal',
     'labelme',
     'caltech'
]



def down_sampling(examples_x,examples_y,remain_sample_size):
    label_list = [0,1,2,3,4]

    # print('examples: ',len(examples))
    example_label = {}
    for (ex_index,example) in enumerate(examples_x):
        if examples_y[ex_index] not in example_label:
            example_label[int(examples_y[ex_index].numpy())] = [example.unsqueeze(0)]
        else:
            example_label[int(examples_y[ex_index].numpy())].append(example.unsqueeze(0))

    # print('example_label: ',example_label)
    final_examples = []
    labels = []
    for label in label_list:
        if label not in example_label: continue
        final_examples += example_label[label][:remain_sample_size] #randonly pick 1 to add, if the down_sampled_size is too small
        labels += [label] * len(example_label[label][:remain_sample_size])
    final_labels=torch.LongTensor(np.array(labels, dtype=int)).view(-1)
    final_examples=torch.cat(final_examples,0)

    print(final_examples.size())
    print(final_labels.size())

    return final_examples,final_labels

def get(logger=None,args=None):

    data={}
    taskcla=[]
    # size=[3,32,32] see arg.image_size and arg.image_channel

    f_name = 'vlcs_'+str(args.ntasks)
    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()
    print('random_sep: ',random_sep)


    for t in range(args.ntasks):
        data[t] = read_celeba(args=args,logger=logger,domain=domains[t],t=t)
        taskcla.append((t,data[t]['ncla']))

    print('taskcla: ',taskcla)

    _,_ = save_statistic(data,taskcla, args)


    return data,taskcla



def save_statistic(data,taskcla,args):
    # label_map = {0:'negative',1:'positive'}
    cls_num_list = []
    data_domain_train = []
    fn = './res/'+args.scenario+'/'+args.task+'/'+args.experiment+'_'+args.approach+'_'+str(args.note)+'statistic.txt'
    with open(fn,'w') as fn_w:
        for t in range(args.ntasks):
            fn_w.writelines(data[t]['name'] + '\t')
            data_label = {}
            data_label_train = {}
            data_domain_train.append(0)

            for c in range(taskcla[t][1]): #clear
                data_label_train[c] = 0

            for s in ['train','valid','test']:

                for c in range(taskcla[t][1]): #clear
                    data_label[c] = 0

                loader = DataLoader(data[t][s], batch_size=1)
                for dat in loader:
                    if args.mtl:
                        images, targets, tasks = dat
                    else:
                        images, targets = dat

                    data_label[targets.item()] += 1

                    if s == 'train':
                        data_label_train[targets.item()] += 1
                        data_domain_train[-1] += 1


                line = ''
                for c in range(taskcla[t][1]):
                    line += str(data_label[c])+'\t'
                fn_w.writelines(line)

            fn_w.writelines('\n')

            cls_num_list.append([data_label_train[i] for i in range(taskcla[t][1])]) #pos and neg
    # os.remove(fn)

    return cls_num_list,data_domain_train

def read_celeba(pc_valid=0.10,args=None,logger=None,domain=None,t=None):
    data={}
    # size=[3, 32, 32] see arg.image_size and arg.image_channel

    data_type = args.data_size

    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]

    # celeba
    dat={}
    #TODO:  one should save the data down since it takes a very long time

    file_name = './dat/vlcs/' + domain + '_binary_celeba/' + str(args.ntasks) + '/'

    if not os.path.isdir(file_name):
        os.makedirs(file_name)

        train_dataset = VLCSLoader(img_dir='./dat/vlcs/'+domain+'/train/',transform=transforms.Compose([transforms.Resize(size=(args.image_size,args.image_size)),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['train'] = train_dataset

        test_dataset = VLCSLoader(img_dir='./dat/vlcs/'+domain+'/test/',transform=transforms.Compose([transforms.Resize(size=(args.image_size,args.image_size)),transforms.ToTensor(),transforms.Normalize(mean,std)]))
        dat['test'] = test_dataset


        # # totally 10 tasks, each tasks 2 classes (whether smiling)
        #
        data={}
        data['name'] = domain
        data['ncla'] = 5


        for s in ['train','test']:
            loader=torch.utils.data.DataLoader(dat[s],batch_size=1) # not shuffle

            data[s]={'x': [],'y': []}

            for image,target in loader:
                label=target.numpy()[0]
                data[s]['x'].append(image)
                data[s]['y'].append(label)

        # # "Unify" and save
        for s in ['train','test']:
            torch.save(data[s]['x'], os.path.join(os.path.expanduser(file_name),'data'+str(t)+s+'x.bin'))
            torch.save(data[s]['y'], os.path.join(os.path.expanduser(file_name),'data'+str(t)+s+'y.bin'))


    # # "Unify" and save
    data = dict.fromkeys(['name','ncla','train','test'])
    for s in ['train','test']:
        data[s]={'x':[],'y':[]}
        data[s]['x']=torch.load(os.path.join(os.path.expanduser(file_name),'data'+str(t)+s+'x.bin'))
        data[s]['y']=torch.load(os.path.join(os.path.expanduser(file_name),'data'+str(t)+s+'y.bin'))
        data[s]['x']=torch.stack(data[s]['x']).view(-1,args.image_channel,args.image_size,args.image_size)
        data[s]['y']=torch.LongTensor(np.array(data[s]['y'],dtype=int)).view(-1)
    data['ncla']=len(np.unique(data['train']['y'].numpy()))
    data['name']=domain



    logger.info(data['name'])

    r=np.arange(data['train']['x'].size(0))
    r=np.array(shuffle(r,random_state=args.data_seed),dtype=int)  # seed set
    nvalid=int(pc_valid*len(r))
    ivalid=torch.LongTensor(r[:nvalid])

    itrain=torch.LongTensor(r[nvalid:])
    data['valid']={}
    data['valid']['x']=data['train']['x'][ivalid].clone()
    data['valid']['y']=data['train']['y'][ivalid].clone()
    data['train']['x']=data['train']['x'][itrain].clone()
    data['train']['y']=data['train']['y'][itrain].clone()
    data['num_train_steps'] = 0

    all_tasks = torch.tensor([t for f in data['train']['y']], dtype=torch.long)


    if args.fewshot:
        data['train']['x'],data['train']['y'] = down_sampling(data['train']['x'],data['train']['y'],args.each_class_remain_sample)
        all_tasks = all_tasks[:args.each_class_remain_sample*args.nclasses]

    if args.distill_loss:
        train_data = InstanceSample(
            data['train']['x'],
            data['train']['y']
            )
    elif args.mtl:
        train_data = TensorDataset(
            data['train']['x'],
            data['train']['y'],
            all_tasks
            )
    else:
        train_data = TensorDataset(
            data['train']['x'],
            data['train']['y']
            )


    data['train']=train_data


    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))

    all_tasks = torch.tensor([t for f in data['valid']['y']], dtype=torch.long)
    if args.dev_data_size > 0:
        # random.Random(args.data_seed).shuffle(eval_data) #unlike dsc, not neccessary to shuffle
        data['valid']['x'] = data['valid']['x'][:args.dev_data_size]
        data['valid']['y'] = data['valid']['y'][:args.dev_data_size]
        all_tasks = all_tasks[:args.dev_data_size]

    if args.distill_loss:
        eval_data = InstanceSample(
            data['valid']['x'],
            data['valid']['y']
            )
    elif args.mtl:
        eval_data = TensorDataset(
            data['valid']['x'],
            data['valid']['y'],
            all_tasks
            )
    else:
        eval_data = TensorDataset(
            data['valid']['x'],
            data['valid']['y']
            )

    data['valid']=eval_data

    logger.info("***** Running Dev *****")
    logger.info("  Num examples = %d", len(eval_data))

    all_tasks = torch.tensor([t for f in data['test']['y']], dtype=torch.long)
    if args.test_data_size > 0:
        # random.Random(args.data_seed).shuffle(test_data)  #unlike dsc, not neccessary to shuffle
        data['test']['x'] = data['test']['x'][:args.test_data_size]
        data['test']['y'] = data['test']['y'][:args.test_data_size]
        all_tasks = all_tasks[:args.test_data_size]

    if args.distill_loss:
        test_data = InstanceSample(
            data['test']['x'],
            data['test']['y']
            )
    elif args.mtl:
        test_data = TensorDataset(
            data['test']['x'],
            data['test']['y'],
            all_tasks
            )
    else:
        test_data = TensorDataset(
            data['test']['x'],
            data['test']['y']
            )

    data['test']=test_data

    logger.info("***** Running testing *****")
    logger.info("  Num examples = %d", len(test_data))


    return data


########################################################################################################################



# customize dataset class

class VLCSLoader(Dataset):
    """Federated EMNIST dataset."""

    def __init__(self,img_dir, transform=None):
        self.transform = transform
        self.size=[227, 227, 3]

        self.x = []
        self.y = []
        for label in os.listdir(img_dir):
            for img_name in os.listdir(img_dir+label):
                im = Image.open(img_dir+label +'/'+img_name)
                np_im = np.array(im)
                torch_img = torch.from_numpy(np_im)
                if list(torch_img.size()) != self.size:
                    torch_img = torch_img.unsqueeze(-1).repeat([1,1,3]) #some are only 1 channel
                self.x.append(torch_img)
                self.y.append(int(label))

        self.x=torch.cat(self.x,0).view(-1,self.size[0],self.size[1],self.size[2])
        self.y=torch.LongTensor(np.array([f for f in self.y],dtype=int)).view(-1).numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        x = self.x[idx]
        y = self.y[idx]

        x = x.data.numpy()
        x = Image.fromarray(x)
        # x = Image.fromarray((x * 255).astype(np.uint8))

        if self.transform:
            x = self.transform(x)
        return x,y



