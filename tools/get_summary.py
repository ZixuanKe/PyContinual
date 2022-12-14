import os
import numpy as np


# method = 'dualprompt_drg_bart_full'

# r1,r2,rL = [],[],[]
# for seed in [2021, 111, 222, 333, 444]:
#     seed = str(seed)
#     path = f'/data/haowei/haowei/data/seq0/seed{seed}'
#     path = os.path.join(path, method)

#     rouge1 = f'progressive_main_{seed}'
#     #rouge2 = f'progressive_rouge2_{seed}'
#     #rougeL = f'progressive_rougeL_{seed}'

#     rouge1 = np.loadtxt(os.path.join(path, rouge1))
#     #rouge2 = np.loadtxt(os.path.join(path, rouge2))
#     #rougeL = np.loadtxt(os.path.join(path, rougeL))
    
#     r1.append(np.average(rouge1[-1, :]))
#     #r2.append(np.average(rouge2[-1, :]))
#     #rL.append(np.average(rougeL[-1, :]))

# print('rouge1', np.std(r1, axis=0), '\n', np.average(r1, axis=0))
# #print('rouge2', np.std(r2, axis=0), '\n', np.average(r2, axis=0))
# #print('rougeL', np.std(rL, axis=0), '\n', np.average(rL, axis=0))

method = 'adapter_hat_asc_roberta_full'

r1,r2,rL = [],[],[]
for seed in [0,1,2]:
    seed = str(seed)
    path = f'./data/seq{seed}/seed2021'
    path = os.path.join(path, method)

    rouge1 = f'progressive_main_2021'
    rouge2 = f'progressive_accuracy_2021'
    #rougeL = f'progressive_rougeL_2021'

    rouge1 = np.loadtxt(os.path.join(path, rouge1))
    rouge2 = np.loadtxt(os.path.join(path, rouge2))
    #rougeL = np.loadtxt(os.path.join(path, rougeL))
    print(rouge1)
    #print(rouge2)
    r1.append(np.average(rouge1[-1, :]))
    r2.append(np.average(rouge2[-1, :]))
    #rL.append(np.average(rougeL[-1, :]))
print(r1)
print('rouge1', np.std(r1, axis=0), '\n', np.average(r1, axis=0))
print('rouge2', np.std(r2, axis=0), '\n', np.average(r2, axis=0))
#print('rougeL', np.std(rL, axis=0), '\n', np.average(rL, axis=0))