import json
import os.path
import random

import jsonlines
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import os


# machine = '/hdd_3/zke4/'
machine = '/HPS/MultiClassSampling/work/zixuan/data/'
#
# ncl_ner_full


asc_datasets = [
        'XuSemEval14_rest',
        'XuSemEval14_laptop',

        'Bing3domains_Speaker',
        'Bing3domains_Router',
        'Bing3domains_Computer',

        'Bing5domains_Nokia6610',
        'Bing5domains_NikonCoolpix4300',
        'Bing5domains_CreativeLabsNomadJukeboxZenXtra40GB',
        'Bing5domains_CanonG3',
        'Bing5domains_ApexAD2600Progressive',

        'Bing9domains_CanonPowerShotSD500',
        'Bing9domains_CanonS100',
        'Bing9domains_DiaperChamp',
        'Bing9domains_HitachiRouter',
        'Bing9domains_ipod',
        'Bing9domains_LinksysRouter',
        'Bing9domains_MicroMP3',
        'Bing9domains_Nokia6600',
        'Bing9domains_Norton']

five_large_datasets = ['yahoo', 'yelp', 'amazon', 'dbpedia', 'agnews']
dialogue_datasets = ['MWOZ_taxi', 'MWOZ_train', 'MWOZ_restaurant', 'MWOZ_hotel', 'MWOZ_attraction', 'sgd_services',
                          'sgd_flights', 'sgd_buses', 'sgd_ridesharing', 'sgd_rentalcars',
                          'sgd_homes', 'sgd_music', 'sgd_events', 'sgd_banks', 'sgd_hotels', 'sgd_calendar',
                          'sgd_media', 'sgd_movies', 'sgd_restaurants', 'sgd_alarm', 'sgd_weather',
                          'sgd_travel', 'sgd_payment', 'sgd_trains',
                          'tma_movie', 'tma_auto', 'tma_restaurant', 'tma_pizza', 'tma_uber', 'tma_coffee',
                          'tmb_hotel', 'tmb_movie', 'tmb_flight', 'tmb_sport',
                          'tmb_restaurant', 'tmb_music', 'tmb_food-ordering'
                          ]
ner_datasets = ['ieer', 'btc', 'gum', 'ritter', 're3d', 'wnut2017', 'wikigold', 'conll2003', 'ontonote']
summerization_datasets = ['nyt', 'stack', 'emails', 'reddit', 'icsi', 'ami']  # current 4 datasets



# ,'adapter_ctr_asc_full','adapter_ctr_mix_full'
# 'l2p_mix_full'




with open('forgetting_rate','w') as fr_file:

    for seq_name in ['ncl_supsup_adapter_ggg_five_large_500_ce_full']:
        seq_location = [
            machine + '/seq0/seed2021/'+seq_name,
            machine + '/seq1/seed2021/'+seq_name,
            machine + '/seq2/seed2021/'+seq_name,
            machine + '/seq3/seed2021/'+seq_name,
            machine + '/seq4/seed2021/'+seq_name
        ]


        if 'sum' in seq_name:

            seq_r1 = []
            seq_r2 = []
            seq_rl = []


            for seq in seq_location:
                r1 = np.loadtxt(os.path.join(seq, 'progressive_rouge1_2021'))
                r2 = np.loadtxt(os.path.join(seq, 'progressive_rouge2_2021'))
                rl = np.loadtxt(os.path.join(seq, 'progressive_rougeL_2021'))

                ntasks = len(r1)

                fr_r1 = []
                fr_r2 = []
                fr_rl = []


                for i in range(ntasks-1): # exlude post=train, if conv_post_sum

                    fr_r1.append(r1[i][i] - r1[ntasks-1][i])
                    fr_r2.append(r2[i][i] - r2[ntasks-1][i])
                    fr_rl.append(rl[i][i] - rl[ntasks-1][i])

                fr_r1 = np.mean(fr_r1)
                fr_r2 = np.mean(fr_r2)
                fr_rl = np.mean(fr_rl)

                seq_r1.append(fr_r1)
                seq_r2.append(fr_r2)
                seq_rl.append(fr_rl)

            seq_r1 = np.mean(seq_r1)
            seq_r2 = np.mean(seq_r2)
            seq_rl = np.mean(seq_rl)

            print('seq_r1, seq_r2, seq_rl: ',seq_r1, seq_r2, seq_rl)

            fr_file.writelines(seq_name + '\t' + str(seq_r1) + '\t' + str(seq_r2) + '\t' + str(seq_rl) + '\n')

        elif 'asc' in seq_name or 'large' in seq_name:

            seq_f1 = []
            seq_acc = []

            for seq in seq_location:
                print('seq: ',seq)
                f1 = np.loadtxt(os.path.join(seq, 'progressive_macro_f1_2021'))
                acc = np.loadtxt(os.path.join(seq, 'progressive_accuracy_2021'))

                ntasks = len(f1)

                fr_f1 = []
                fr_acc = []

                for i in range(ntasks - 1):

                    # TODO: there is some special things for ASC in CAT, because it tasks so long
                    # if i <= 16 and 'asc' in seq_name and 'cat' in seq_name:
                    #     fr_f1.append(f1[16][i] - f1[ntasks - 1][i])
                    #     fr_acc.append(acc[16][i] - acc[ntasks - 1][i])
                    # else:
                    fr_f1.append(f1[i][i] - f1[ntasks - 1][i])
                    fr_acc.append(acc[i][i] - acc[ntasks - 1][i])

                print('fr_f1, fr_acc: ', fr_f1, fr_acc)

                fr_f1 = np.mean(fr_f1)
                fr_acc = np.mean(fr_acc)

                seq_f1.append(fr_f1)
                seq_acc.append(fr_acc)

            print('seq_f1, seq_acc: ', seq_f1, seq_acc)

            seq_f1 = np.mean(seq_f1)
            seq_acc = np.mean(seq_acc)


            fr_file.writelines(seq_name + '\t' + str(seq_f1) + '\t' + str(seq_acc) + '\n')

        elif 'ner' in seq_name:
            seq_f1 = []

            for seq in seq_location:
                f1 = np.loadtxt(os.path.join(seq, 'progressive_f1_2021'))

                ntasks = len(f1)

                fr_f1 = []

                for i in range(ntasks - 1):
                    fr_f1.append(f1[i][i] - f1[ntasks - 1][i])

                fr_f1 = np.mean(fr_f1)

                seq_f1.append(fr_f1)

            seq_f1 = np.mean(seq_f1)

            print('seq_f1: ', seq_f1)

            fr_file.writelines(seq_name + '\t' + str(seq_f1) + '\n')


        elif 'dialogue' in seq_name:
            seq_bleu = []

            for seq in seq_location:
                bleu = np.loadtxt(os.path.join(seq, 'progressive_bleu_2021'))

                ntasks = len(bleu)

                fr_bleu = []

                for i in range(ntasks - 1):
                    fr_bleu.append(bleu[i][i] - bleu[ntasks - 1][i])

                fr_bleu = np.mean(fr_bleu)

                seq_bleu.append(fr_bleu)

            seq_bleu = np.mean(seq_bleu)

            print('seq_bleu: ', seq_bleu)

            fr_file.writelines(seq_name + '\t' + str(seq_bleu) + '\n')



        elif 'mix' in seq_name:
            seq_r1 = []
            seq_r2 = []
            seq_rl = []
            seq_bleu = []
            seq_f1 = []
            seq_asc_f1 = []
            seq_asc_acc = []
            seq_five_large_f1 = []
            seq_five_large_acc = []


            # Summerization
            for seq in seq_location:

                seq_id = int(seq.split('seq')[1][0])
                with open('mix_seq', 'r') as f_random_seq:
                    random_sep = f_random_seq.readlines()[seq_id].split()

                sum_id = [d_id for d_id,d in enumerate(random_sep) if d in summerization_datasets]
                asc_id = [d_id for d_id,d in enumerate(random_sep) if d in asc_datasets]
                five_large_id = [d_id for d_id,d in enumerate(random_sep) if d in five_large_datasets]
                dialogue_id = [d_id for d_id,d in enumerate(random_sep) if d in dialogue_datasets]
                ner_id = [d_id for d_id,d in enumerate(random_sep) if d in ner_datasets]

                r1 = np.loadtxt(os.path.join(seq, 'progressive_rouge1_2021'))
                r2 = np.loadtxt(os.path.join(seq, 'progressive_rouge2_2021'))
                rl = np.loadtxt(os.path.join(seq, 'progressive_rougeL_2021'))

                ntasks = len(r1)

                fr_r1 = []
                fr_r2 = []
                fr_rl = []

                for i in range(ntasks-1): # exlude post=train, if conv_post_sum
                    if i in sum_id:
                        fr_r1.append(r1[i][i] - r1[ntasks-1][i])
                        fr_r2.append(r2[i][i] - r2[ntasks-1][i])
                        fr_rl.append(rl[i][i] - rl[ntasks-1][i])

                fr_r1 = np.mean(fr_r1)
                fr_r2 = np.mean(fr_r2)
                fr_rl = np.mean(fr_rl)

                seq_r1.append(fr_r1)
                seq_r2.append(fr_r2)
                seq_rl.append(fr_rl)

            seq_r1 = np.mean(seq_r1)
            seq_r2 = np.mean(seq_r2)
            seq_rl = np.mean(seq_rl)

            print('seq_r1, seq_r2, seq_rl: ',seq_r1, seq_r2, seq_rl)

            fr_file.writelines(str(seq_r1) + '\t')


            # ASC
            for seq in seq_location:
                seq_id = int(seq.split('seq')[1][0])
                with open('mix_seq', 'r') as f_random_seq:
                    random_sep = f_random_seq.readlines()[seq_id].split()

                sum_id = [d_id for d_id,d in enumerate(random_sep) if d in summerization_datasets]
                asc_id = [d_id for d_id,d in enumerate(random_sep) if d in asc_datasets]
                five_large_id = [d_id for d_id,d in enumerate(random_sep) if d in five_large_datasets]
                dialogue_id = [d_id for d_id,d in enumerate(random_sep) if d in dialogue_datasets]
                ner_id = [d_id for d_id,d in enumerate(random_sep) if d in ner_datasets]

                f1 = np.loadtxt(os.path.join(seq, 'progressive_macro_f1_2021'))
                acc = np.loadtxt(os.path.join(seq, 'progressive_accuracy_2021'))

                ntasks = len(f1)

                fr_f1 = []
                fr_acc = []

                for i in range(ntasks - 1):
                    if i in asc_id:
                        fr_f1.append(f1[i][i] - f1[ntasks - 1][i])
                        fr_acc.append(acc[i][i] - acc[ntasks - 1][i])

                fr_f1 = np.mean(fr_f1)
                fr_acc = np.mean(fr_acc)

                seq_asc_f1.append(fr_f1)
                seq_asc_acc.append(fr_acc)

            seq_asc_f1 = np.mean(seq_asc_f1)
            seq_asc_acc = np.mean(seq_asc_acc)

            print('seq_f1, seq_acc: ', seq_asc_f1, seq_asc_acc)

            fr_file.writelines(str(seq_asc_f1) + '\t')

            # Five Lager
            for seq in seq_location:
                seq_id = int(seq.split('seq')[1][0])
                with open('mix_seq', 'r') as f_random_seq:
                    random_sep = f_random_seq.readlines()[seq_id].split()

                sum_id = [d_id for d_id,d in enumerate(random_sep) if d in summerization_datasets]
                asc_id = [d_id for d_id,d in enumerate(random_sep) if d in asc_datasets]
                five_large_id = [d_id for d_id,d in enumerate(random_sep) if d in five_large_datasets]
                dialogue_id = [d_id for d_id,d in enumerate(random_sep) if d in dialogue_datasets]
                ner_id = [d_id for d_id,d in enumerate(random_sep) if d in ner_datasets]

                f1 = np.loadtxt(os.path.join(seq, 'progressive_macro_f1_2021'))
                acc = np.loadtxt(os.path.join(seq, 'progressive_accuracy_2021'))

                ntasks = len(f1)

                fr_f1 = []
                fr_acc = []

                for i in range(ntasks - 1):
                    if i in five_large_id:
                        fr_f1.append(f1[i][i] - f1[ntasks - 1][i])
                        fr_acc.append(acc[i][i] - acc[ntasks - 1][i])

                fr_f1 = np.mean(fr_f1)
                fr_acc = np.mean(fr_acc)

                seq_five_large_f1.append(fr_f1)
                seq_five_large_acc.append(fr_acc)

            seq_five_large_f1 = np.mean(seq_five_large_f1)
            seq_five_large_acc = np.mean(seq_five_large_acc)

            print('seq_f1, seq_acc: ', seq_five_large_f1, seq_five_large_acc)

            fr_file.writelines(str(seq_five_large_f1) + '\t')


            # Dialogue
            for seq in seq_location:
                seq_id = int(seq.split('seq')[1][0])
                with open('mix_seq', 'r') as f_random_seq:
                    random_sep = f_random_seq.readlines()[seq_id].split()

                sum_id = [d_id for d_id,d in enumerate(random_sep) if d in summerization_datasets]
                asc_id = [d_id for d_id,d in enumerate(random_sep) if d in asc_datasets]
                five_large_id = [d_id for d_id,d in enumerate(random_sep) if d in five_large_datasets]
                dialogue_id = [d_id for d_id,d in enumerate(random_sep) if d in dialogue_datasets]
                ner_id = [d_id for d_id,d in enumerate(random_sep) if d in ner_datasets]
                bleu = np.loadtxt(os.path.join(seq, 'progressive_bleu_2021'))

                ntasks = len(bleu)

                fr_bleu = []

                for i in range(ntasks - 1):
                    if i in dialogue_id:
                        fr_bleu.append(bleu[i][i] - bleu[ntasks - 1][i])

                fr_bleu = np.mean(fr_bleu)

                seq_bleu.append(fr_bleu)

            seq_bleu = np.mean(seq_bleu)

            print('seq_bleu: ', seq_bleu)

            fr_file.writelines(str(seq_bleu) + '\t')


            #NER
            for seq in seq_location:
                seq_id = int(seq.split('seq')[1][0])
                with open('mix_seq', 'r') as f_random_seq:
                    random_sep = f_random_seq.readlines()[seq_id].split()

                sum_id = [d_id for d_id,d in enumerate(random_sep) if d in summerization_datasets]
                asc_id = [d_id for d_id,d in enumerate(random_sep) if d in asc_datasets]
                five_large_id = [d_id for d_id,d in enumerate(random_sep) if d in five_large_datasets]
                dialogue_id = [d_id for d_id,d in enumerate(random_sep) if d in dialogue_datasets]
                ner_id = [d_id for d_id,d in enumerate(random_sep) if d in ner_datasets]
                f1 = np.loadtxt(os.path.join(seq, 'progressive_f1_2021'))

                ntasks = len(f1)

                fr_f1 = []

                for i in range(ntasks - 1):
                    if i in ner_id:
                        fr_f1.append(f1[i][i] - f1[ntasks - 1][i])

                fr_f1 = np.mean(fr_f1)

                seq_f1.append(fr_f1)

            seq_f1 = np.mean(seq_f1)

            print('seq_f1: ', seq_f1)

            fr_file.writelines(str(seq_f1) + '\n')
