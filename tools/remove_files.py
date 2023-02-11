import sys
import os
from os import listdir
import shutil
#
# machine = '/hdd_3/zke4/'
machine = '/HPS/MultiClassSampling/work/zixuan/data/'

# system = 'supsup_pool_adapter_ggg_ner_200_ce_full'
#
machines = ['/HPS/MultiClassSampling/work/zixuan/data/seq0/seed2021/',
'/HPS/MultiClassSampling/work/zixuan/data/seq1/seed2021/',
'/HPS/MultiClassSampling/work/zixuan/data/seq2/seed2021/',
'/HPS/MultiClassSampling/work/zixuan/data/seq3/seed2021/',
'/HPS/MultiClassSampling/work/zixuan/data/seq4/seed2021/']



seqs = ['seq0/','seq1/','seq2/','seq3/','seq4/']
seeds = ['seed2021/','seed111/','seed222/','seed333/']

# seqs = ['seq0/']
# seeds = ['seed222']

# machines = ['/hdd_3/zke4/seq0/seed2021/','/hdd_3/zke4/seq1/seed2021/','/hdd_3/zke4/seq2/seed2021/','/hdd_3/zke4/seq3/seed2021/']


# machines = ['/hdd_3/zke4/' + seq + seed for seq in seqs for seed in seeds]



paths = []



def remove_file(path):
    for item in os.listdir(path):
        if item.endswith(".bin"):
            # if  "fisher" in item:
            os.remove(path + '/' + item)
            print('remove: ' + path + '/'+ item)

for machine in machines:
    if os.path.isdir(machine):
        systems = os.listdir(machine)

        for system in systems:
            paths += [machine + system]

for path in paths:
    if os.path.isdir(path):
        test = os.listdir(path)
        for dir in test:
            path_dir = path + '/' + dir
            if os.path.isdir(path_dir):
                remove_file(path_dir)
                for item in os.listdir(path + '/' + dir):
                    path_item = path + '/' + dir + '/'+item
                    print('path_item: ', path_item)
                    if os.path.isdir(path_item):
                        remove_file(path_item)

