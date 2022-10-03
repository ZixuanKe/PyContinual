import sys
import os
from os import listdir
import shutil
#
# machine = '/hdd_3/zke4/'
machine = '/HPS/MultiClassSampling/work/zixuan/data/'
system = 'ncl_supsup_adapter_ggg_five_large_500_ce_full'

paths = [
    machine + "seq0/seed2021/" + system,
    machine + "seq4/seed2021/"+system,
    machine + "seq3/seed2021/"+system,
    machine + "seq2/seed2021/"+system,
    machine + "seq1/seed2021/"+system
]

for path in paths:
    test = os.listdir(path)
    for dir in test:
        if os.path.isdir(path + '/' + dir):
            for item in os.listdir(path + '/' + dir):
                if item.endswith(".bin") and not "adapter" in item:
                # if  "fisher" in item:
                    os.remove(path + '/' + dir + '/'+item)
                    print('remove: ' + path + '/' + dir + '/'+item)








# folder = '/HPS/MultiClassSampling/work/zixuan/dataset_cache/'
# for filename in os.listdir(folder):
#     file_path = os.path.join(folder, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print('Failed to delete %s. Reason: %s' % (file_path, e))