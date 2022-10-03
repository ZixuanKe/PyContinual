
import json
import os.path
import random

import jsonlines
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer
import re
import json
from collections import defaultdict
from tqdm import tqdm
from tabulate import tabulate
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
import pprint
from termcolor import colored

moz_path = '/sdb/zke4/data_dialogue'
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large', use_fast=not False)
pp = pprint.PrettyPrinter(indent=4)


def get_data_loaders(tokenizer, test=False):
    """ Prepare the dataset for training and evaluation """
    aggregate = get_datasets(dataset_list='MWOZ',setting='single',verbose=False,develop=False)

    if(test):
        datasets = {"test":{}}
    else:
        datasets = {"train":{}, "dev": {}, "test":{}}

    for split in datasets.keys():
        for tasks_id, task in aggregate["BYDOMAIN"][split].items():
            datasets[split][tasks_id] = get_NLG_from_dial(task,tasks_id,tokenizer)


    task_id_train = set(task_id for task_id, dataset_task in datasets["train"].items())
    task_id_dev = set(task_id for task_id, dataset_task in datasets["dev"].items())
    task_id_test = set(task_id for task_id, dataset_task in datasets["test"].items())
    common_task_id = list(task_id_train & task_id_dev & task_id_test)

    ### LOGGING SOME INFORMATION ABOUT THE TASKS
    print(f"Tasks: {common_task_id}")
    print(f"Num of Tasks {len(common_task_id)}")
    task = defaultdict(lambda:defaultdict(str))
    for split in ["train","dev","test"]:
        for task_id, dataset_task in datasets[split].items():
            task[task_id][split] = len(dataset_task)
    table = []
    for dom_name, split_len in task.items():
        table.append({"dom":dom_name, "train":split_len["train"], "dev":split_len["dev"], "test":split_len["test"]})
    print(tabulate(table, headers="keys"))

    common_domain = ["['MWOZ_taxi']", "['MWOZ_hotel']", "['MWOZ_train']", "['MWOZ_restaurant']", "['MWOZ_attraction']"]
    for domain_id,domain in enumerate(['/MWOZ_taxi/', '/MWOZ_train/', '/MWOZ_restaurant/', '/MWOZ_hotel/', '/MWOZ_attraction/']):
        for split in ["train","dev","test"]:
            if split == 'dev':
                split_ = 'validation'
            else:
                split_ = split
            with jsonlines.open(moz_path + domain + split_ + '.jsonl', 'w') as f_json:
                for instance in datasets[split][common_domain[domain_id]]:
                    f_json.write(instance)



    # train_loaders = {}
    # valid_loaders = {}
    # train_datasets = {}
    # val_datasets = {}
    # if(args.continual):
    #     if(not test):
    #         for task_id, dataset_task in datasets["train"].items():
    #             if(task_id in common_task_id):
    #                 train_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.train_batch_size, shuffle=True,collate_fn=partial(collate_fn_, tokenizer=tokenizer))
    #                 train_datasets[task_id] = dataset_task
    #         for task_id, dataset_task in datasets["dev"].items():
    #             if(task_id in common_task_id):
    #                 valid_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.valid_batch_size, shuffle=False,collate_fn=partial(collate_fn_, tokenizer=tokenizer))
    #                 val_datasets[task_id] = dataset_task
    # elif(args.multi):
    #     if(not test):
    #         dataset_train = []
    #         for task_id, dataset_task in datasets["train"].items():
    #             if(task_id in common_task_id):
    #                 dataset_train += dataset_task
    #         train_loaders = DataLoader(DatasetTrain(dataset_train), batch_size=args.train_batch_size, shuffle=True,collate_fn=partial(collate_fn_, tokenizer=tokenizer))
    #
    #         dataset_dev = []
    #         for task_id, dataset_task in datasets["dev"].items():
    #             if(task_id in common_task_id):
    #                 dataset_dev += dataset_task
    #         valid_loaders = DataLoader(DatasetTrain(dataset_dev), batch_size=args.valid_batch_size, shuffle=False,collate_fn=partial(collate_fn_, tokenizer=tokenizer))
    #
    # temp_list = []
    # for task_id, dataset_task in datasets["test"].items():
    #     if(task_id in common_task_id):
    #         temp_list.append(dataset_task)
    # test_datasets = sum(temp_list,[])
    # test_loaders = DataLoader(DatasetTrain(sum(temp_list,[])), batch_size=args.valid_batch_size, shuffle=False,collate_fn=partial(collate_fn_, tokenizer=tokenizer))
    #
    # ### THIS IS JUST FOR CHECKING DUPLICATE DIALOGUES
    # testing_dict = defaultdict(list)
    # for idx_b, batch in tqdm(enumerate(test_loaders),total=len(test_loaders)):
    #     for (d_id, t_id, ta_id) in zip(batch["dial_id"],batch["turn_id"],batch["task_id"]):
    #         if(f'{d_id}_{t_id}_{ta_id}' not in testing_dict):
    #             testing_dict[f'{d_id}_{t_id}_{ta_id}'].append(1)
    #         else:
    #             print(f'{d_id}_{t_id}_{ta_id}')

    # return train_loaders, valid_loaders, test_loaders, (train_datasets,val_datasets,test_datasets)


def get_datasets(dataset_list=['SGD'],setting="single",verbose=False,develop=False):

    table = []
    train = []
    dev = []
    test = []

    if("MWOZ" in dataset_list):
        print("LOAD MWOZ")
        train_MWOZ, dev_MWOZ,test_MWOZ = preprocessMWOZ(develop=develop)
        if(verbose):
            print_sample(train_MWOZ,2)
            input()
        n_domain, n_intent, n_turns, _ = get_domains_slots(train_MWOZ)
        table.append({"Name":"MWOZ","Trn":len(train_MWOZ),"Val":len(dev_MWOZ),"Tst":len(test_MWOZ),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
        train += train_MWOZ
        dev += dev_MWOZ
        test += test_MWOZ


    n_domain, n_intent, n_turns, services = get_domains_slots(train)
    table.append({"Name":"TOT","Trn":len(train),"Val":len(dev),"Tst":len(test),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
    test = filter_services(test,services) ## Remove dialogue with API not present in the test set
    dev = filter_services(dev,services) ## Remove dialogue with API not present in the test set
    n_domain, n_intent, n_turns, services = get_domains_slots(train)
    if(verbose):
        for inten in services:
            print(inten)
        input()
    table.append({"Name":"TOT","Trn":len(train),"Val":len(dev),"Tst":len(test),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
    print(tabulate(table, headers="keys"))

    return {"TOTAL":{"train":train,"dev":dev,"test":test},
               "BYDOMAIN":{"train":split_by_domain(train,setting),
                            "dev":split_by_domain(dev,setting),
                            "test":split_by_domain(test,setting)}
                            }


def split_by_domain(data,setting):
    data_by_domain = defaultdict(list)
    if setting=="single":
        for dial in data:
            if(len(dial["services"])==1):
                data_by_domain[str(sorted(dial["services"]))].append(dial)
        print("SINGLE DOMAIN:",len(data_by_domain.keys()))

    elif setting=="multi":
        data_by_domain = defaultdict(list)
        for dial in data:
            data_by_domain[str(sorted(dial["services"]))].append(dial)
        print("ALL DOMAIN:",len(data_by_domain.keys()))
    else:
        print("choose a setting: --setting single or --setting multi")
        exit(1)
    # for d,v in sorted(data_by_domain.items() ,  key=lambda x: len (eval(x[0]))):
    #     print(d)
    return dict(sorted(data_by_domain.items() ,  key=lambda x: len (eval(x[0]))))



def filter_services(data,serv):
    filtered_dialogue = []
    for dial in data:
        flag_temp = True
        for turn in dial['dialogue']:
            if(turn["spk"]=="API"):
                for s in turn["service"]:
                    if s not in serv:
                        flag_temp = False
        if(flag_temp):
            filtered_dialogue.append(dial)
    return filtered_dialogue


def print_sample(data,num):
    color_map = {"USER":"blue","SYSTEM":"magenta","API":"red","API-OUT":"green"}
    for i_d, dial in enumerate(random.sample(data,len(data))):
        print(f'ID:{dial["id"]}')
        print(f'Services:{dial["services"]}')
        for turn in dial['dialogue']:
            print(colored(f'{turn["spk"]}:',color_map[turn["spk"]])+f' {turn["utt"]}')
        if i_d == num: break


def get_domains_slots(data):
    services = set()
    intent = set()
    len_dialogue = []
    for dial in data:
        for s in dial["services"]:
            services.add(s)
        len_dialogue.append(len([0 for t in dial['dialogue'] if t["spk"] in ["USER","SYSTEM"]]))
        for turn in dial['dialogue']:
            if(turn["spk"]=="API"):
                for s in turn["service"]:
                    if(" " in s or len(s)==1):
                        print(s)
                        print(turn)
                        input()
                    intent.add(s)
    print("Domain",len(services))
    print("Intent",len(intent))
    print("Avg. Turns",np.mean(len_dialogue))
    return len(services), len(intent), np.mean(len_dialogue), intent


def get_NLG_from_dial(data,task_id,tokenizer):
    dialogues = []
    utt_len = []
    hist_len = []
    for dial in data:
        plain_history = []
        latest_API_OUT = "API-OUT: "
        for idx_t, t in enumerate(dial['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9":
                continue
            if(t['spk']=="USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif(t['spk']=="API-OUT"):
                latest_API_OUT = f"{t['utt'].strip()}"
            elif((t['spk'] == "SYSTEM") and idx_t!=0 and t["utt"]!= ""):
                if(latest_API_OUT != ""):
                    dialogues.append({"history":latest_API_OUT,
                                    "reply":f'{t["utt"].strip()} {tokenizer.eos_token}',
                                    "history_reply": latest_API_OUT + f'[SOS]{t["utt"].strip()} {tokenizer.eos_token}',
                                    "spk":t["spk"],
                                    "dataset":t["dataset"],
                                    "dial_id":t["id"],
                                    "turn_id":t["turn_id"],
                                    "task_id":task_id})
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
                latest_API_OUT = ""

    for d in random.sample(dialogues,len(dialogues)):
        pp.pprint(d)
        break
    print()
    input()
    return dialogues

def preprocessMWOZ(develop=False):
    data = []
    dialogue = json.load(open(moz_path+"/multiwoz/data/MultiWOZ_2.2/data.json"))
    for i_d, (d_idx, d) in tqdm(enumerate(dialogue.items()),total=len(dialogue.items())):
        dial = {"id":d_idx, "services": get_domains(d['goal']), "dataset":"MWOZ"}
        if "MWOZ_police" in dial["services"] or "MWOZ_hospital" in dial["services"] or "MWOZ_bus" in dial["services"]: continue
        turns =[]
        dst_prev = {}
        for t_idx, t in enumerate(d['log']):
            if(t_idx % 2 ==0):
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"USER","utt":t["text"]})
                # print("USER",t["text"])
                str_API_ACT = ""
                if "dialog_act" in t:
                    intents_act = set()
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Request" in k:
                            str_API_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"',"'")
                                    str_API_ACT += f'{s.lower()}="{v}",'
                                    # str_API_ACT += f'{k.lower().replace('-','_')}.{s.lower()} = "{v}" '
                                    intents_act.add(k.lower().replace('-','_'))
                            if(str_API_ACT[-1]==","):
                                str_API_ACT = str_API_ACT[:-1]
                            str_API_ACT += ") "
                # print("API", str_API)
            else:
                dst_api = get_value_dst(t["metadata"])
                str_API = ""
                intents = set()
                for k,slt in dst_api.items():
                    str_API += f"{k.lower().replace('-','_')}("
                    for (s,v) in slt:
                        if len(v)!= 0:
                            v = v[0].replace('"',"'")
                            str_API += f'{s.lower()}="{v}",'
                            intents.add(k.lower().replace('-','_'))
                    if(len(str_API)>0 and str_API[-1]==","):
                        str_API = str_API[:-1]
                    str_API += ") "
                if(str_API==""):
                    turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API","utt":str_API_ACT,"service":list(intents_act)})
                    # print("API",str_API_ACT)
                else:
                    turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API","utt":str_API,"service":list(intents)})
                    # print("API", str_API)

                ## API RETURN
                str_ACT = ""
                if "dialog_act" in t:
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Recommend" in k or "Booking-Book" in k or "-Select" in k:
                            str_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"',"'")
                                    str_ACT += f'{s.lower()}="{v}",'
                                    # str_ACT += f'{k.lower().replace("-",".")}.{s.lower()} = "{v}" '
                            if(str_ACT[-1]==","):
                                str_ACT = str_ACT[:-1]
                            str_ACT += ") "
                        if "Booking-NoBook" in k:
                            # str_ACT += f'{k.lower().replace("-",".")} '
                            str_ACT += f"{k.lower().replace('-','_')}() "

                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API-OUT","utt":str_ACT,"service":None})
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"SYSTEM","utt":t["text"]})
        dial["dialogue"] = turns
        data.append(dial)
        if(develop and i_d==10): break


    split_id_dev, split_id_test = loadCSV("val"), loadCSV("test")

    train_data, dev_data, test_data = [], [], []

    for dial in data:
        if dial["id"] in split_id_dev:
           dev_data.append(dial)
        elif dial["id"] in split_id_test:
           test_data.append(dial)
        else:
           train_data.append(dial)
    return train_data, dev_data, test_data



def get_domains(goal):
    dom = []
    for d, g in goal.items():
        if(len(g)!=0) and d!= "message" and d!= "topic":
            dom.append("MWOZ_"+d)
    return dom

def loadCSV(split):
    split_id = []
    with open(moz_path+f"/multiwoz/data/MultiWOZ_2.1/{split}ListFile.txt") as f:
        for l in f:
            split_id.append(l.replace("\n",""))
    return split_id

def get_value_dst(DST):
    active_dst = defaultdict(list)
    for k,v in DST.items():
        for k_s, v_s in v['semi'].items():
            if(len(v_s)!=0):
                active_dst[k].append([k_s, v_s])
        for k_s, v_s in v['book'].items():
            if(len(v_s)!=0 and k_s != "booked"):
                active_dst[k].append([k_s, v_s])
    return active_dst



get_data_loaders(tokenizer)
