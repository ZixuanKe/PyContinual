
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
import glob

moz_path = '/sdb/zke4/data_dialogue'
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large', use_fast=not False)
pp = pprint.PrettyPrinter(indent=4)
allowed_ACT_list = ["INFORM","CONFIRM","OFFER","NOTIFY_SUCCESS","NOTIFY_FAILURE","INFORM_COUNT"]

DOMAINS =["uber", "movie", "restaurant", "coffee", "pizza", "auto", "sport", "flight", "food-ordering", "hotel", "music"]


def get_data_loaders(tokenizer, test=False):
    """ Prepare the dataset for training and evaluation """
    aggregate = get_datasets(dataset_list=['TM19','TM20'],setting='single',verbose=False,develop=False)

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

    common_domain = [
        "['tma_movie']", "['tma_auto']", "['tma_restaurant']", "['tma_pizza']", "['tma_uber']",
        "['tma_coffee']", "['tmb_hotel']", "['tmb_movie']", "['tmb_flight']", "['tmb_sport']",
        "['tmb_restaurant']", "['tmb_music']", "['tmb_food-ordering']"
    ]
    for domain_id,domain in enumerate(
            [
                '/tma_movie/', '/tma_auto/', '/tma_restaurant/', '/tma_pizza/', '/tma_uber/',
                '/tma_coffee/', '/tmb_hotel/', '/tmb_movie/', '/tmb_flight/', '/tmb_sport/',
                '/tmb_restaurant/', '/tmb_music/', '/tmb_food-ordering/'
            ]):
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


def get_datasets(dataset_list=['TM19','TM20'],setting="single",verbose=False,develop=False):

    table = []
    train = []
    dev = []
    test = []


    if("TM19" in dataset_list):
        print("LOAD TM19")
        train_TM19, dev_TM19, test_TM19 = preprocessTM2019(develop=develop)
        if(verbose):
            print_sample(train_TM19,2)
            input()
        n_domain, n_intent, n_turns, _ = get_domains_slots(train_TM19)
        table.append({"Name":"TM19","Trn":len(train_TM19),"Val":len(dev_TM19),"Tst":len(test_TM19),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
        train += train_TM19
        dev += dev_TM19
        test += test_TM19

    if("TM20" in dataset_list):
        print("LOAD TM20")
        train_TM20, dev_TM20, test_TM20 = preprocessTM2020(develop=develop)
        if(verbose):
            print_sample(train_TM20,2)
            input()
        n_domain, n_intent, n_turns, _ = get_domains_slots(train_TM20)
        table.append({"Name":"TM20","Trn":len(train_TM20),"Val":len(dev_TM20),"Tst":len(test_TM20),"Dom":n_domain,"Int":n_intent,"Tur":n_turns})
        train += train_TM20
        dev += dev_TM20
        test += test_TM20


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

def remove_numbers_from_string(s):
    numbers = re.findall(r'\d+', s)
    for n in numbers:
        s = s.replace(n,"")
    return s

def create_API_str(API_struct):
    str_API = ""
    for k,v in API_struct.items():
        str_API += f"{k}("
        for arg, val in v.items():
            val = val.replace('"',"'")
            str_API += f'{arg}="{val}",'

        str_API = str_API[:-1]
        str_API += ") "
    return str_API

def prepocess_API(frame):
    if(frame==""):
        return "",None,[]
    else:
        API_struct = defaultdict(lambda: defaultdict(str))
        services = set()
        for s in frame:
            parsed = remove_numbers_from_string(s["annotations"][-1]["name"]).split(".")
            API_struct[parsed[0]]["_".join(parsed[1:])] = s["text"]
            services.add(parsed[0])
        # print(API_struct)
        services = [s.strip() for s in services]
        return create_API_str(API_struct), API_struct, list(services)



def fix_turn(turns):
    for i_t, t in enumerate(turns):
        if(t['spk']=="API-OUT" or t['spk']=="API"):
            t['utt'], t['n_struct'], t['service'] = prepocess_API(t["struct"])
    if len(turns)>0 and turns[0]['spk'] == "API-OUT":
        turns = turns[1:]

    if len(turns)>0 and turns[-1]['spk'] == "API":
        turns = turns[:-1]
    new_turns = []
    for i_t, t in enumerate(turns):
        if(t['spk']=="API-OUT" and t['utt']!=""):
            if turns[i_t-1]['utt']=="":
                new_turns[-1]["utt"] = list(turns[i_t]['n_struct'].keys())[0]+"()"
                new_turns[-1]["service"] = [list(turns[i_t]['n_struct'].keys())[0]]


        new_turns.append(turns[i_t])
    return new_turns

def get_data(dialogue,year,develop=False):
    data = []
    for i_d,d in tqdm(enumerate(dialogue),total=len(dialogue)):

        #### GET THE DOMAIN OF THE DIALOGUE
        flag = True
        serv = ""
        for dom in DOMAINS:
            if(dom in d["instruction_id"]):
                serv = f"{dom}"
                flag = False
            elif("hungry" in d["instruction_id"] or
                "dinner" in d["instruction_id"] or
                "lunch" in d["instruction_id"] or
                "dessert" in d["instruction_id"]):
                serv = f"restaurant"
                flag = False
            elif("nba" in d["instruction_id"] or
                 "mlb" in d["instruction_id"] or
                 "epl" in  d["instruction_id"] or
                 "mls" in d["instruction_id"] or
                 "nfl" in d["instruction_id"] ):
                serv = f"sport"
                flag = False
        if(flag): print(d["instruction_id"])
        ####

        dial = {"id":d["conversation_id"].strip(), "services": [f"TM{year}_"+serv], "dataset":f"TM{year}"}
        turns =[]
        for t_idx, t in enumerate(d["utterances"]):
            if(t["speaker"]=="USER"):
                if(len(turns)!=0 and turns[-1]['spk']=="API"):
                    turns[-2]["utt"] += " "+t["text"]
                    if "segments" in t and type(turns[-1]["struct"])==list:
                        turns[-1]["struct"] += t["segments"]
                    elif("segments" in t and type(turns[-1]["struct"])==str):
                        turns[-1]["struct"] = t["segments"]
                    else:
                        turns[-1]["struct"] = ""
                else:
                    turns.append({"dataset":f"TM{year}","id":d["conversation_id"].strip(),"turn_id":t_idx,"spk":t["speaker"],"utt":t["text"]})
                    turns.append({"dataset":f"TM{year}","id":d["conversation_id"].strip(),"turn_id":t_idx,"spk":"API","utt":"","struct":t["segments"] if "segments" in t else "","service":[]})
            else:
                if(len(turns)!=0 and turns[-1]['spk']=="SYSTEM"):
                    turns[-1]["utt"] += " "+t["text"]
                    if "segments" in t and type(turns[-2]["struct"])==list:
                        turns[-2]["struct"] += t["segments"]
                    elif("segments" in t and type(turns[-2]["struct"])==str):
                        turns[-2]["struct"] = t["segments"]
                    else:
                        turns[-2]["struct"] = ""
                else:
                    turns.append({"dataset":f"TM{year}","id":d["conversation_id"].strip(),"turn_id":t_idx,"spk":"API-OUT","utt":"","struct":t["segments"] if "segments" in t else "","service":[]})
                    turns.append({"dataset":f"TM{year}","id":d["conversation_id"].strip(),"turn_id":t_idx,"spk":"SYSTEM","utt":t["text"]})


        dial["dialogue"] = fix_turn(turns)
        data.append(dial)
        if(develop and i_d==10): break
    return data

def preprocessTM2019(develop=False):
    dialogue = json.load(open(moz_path+f"/Taskmaster/TM-1-2019/woz-dialogs.json"))
    print('dialogue: ',dialogue)

    data = get_data(dialogue,"A",develop)

    data_by_domain = defaultdict(list)
    for dial in data:
        if(len(dial["services"])==1):
            data_by_domain[str(sorted(dial["services"]))].append(dial)

    data_by_domain_new = defaultdict(list)
    for dom, data in data_by_domain.items():
        train_data, dev_data, test_data = np.split(data, [int(len(data)*0.8), int(len(data)*0.9)])
        data_by_domain_new[str(sorted([remove_numbers_from_string(s) for s in eval(dom)]))].append([train_data, dev_data, test_data])

    train_data, valid_data, test_data = [], [], []
    table = []
    for dom, list_of_split in data_by_domain_new.items():
        train, valid, test = [], [], []
        for [tr,va,te] in list_of_split:
            train += rename_service_dialogue(tr,dom)
            valid += rename_service_dialogue(va,dom)
            test += rename_service_dialogue(te,dom)
        table.append({"dom":dom, "train":len(train), "valid":len(valid), "test":len(test)})
        train_data += train
        valid_data += valid
        test_data += test
    print(tabulate(table, headers="keys"))

    return train_data,valid_data,test_data


def rename_service_dialogue(dial_split,name):
    new_dial = []
    for dial in dial_split:
        dial["services"] = eval(name)
        new_dial.append(dial)
    return new_dial

def preprocessTM2020(develop=False):
    data = []
    for f in glob.glob(moz_path+f"/Taskmaster/TM-2-2020/data/*.json"):
        dialogue = json.load(open(f))
        data += get_data(dialogue,"B",develop)

    data_by_domain = defaultdict(list)
    for dial in data:
        if(len(dial["services"])==1):
            data_by_domain[str(sorted(dial["services"]))].append(dial)

    data_by_domain_new = defaultdict(list)
    for dom, data in data_by_domain.items():
        train_data, dev_data, test_data = np.split(data, [int(len(data)*0.8), int(len(data)*0.9)])
        data_by_domain_new[str(sorted([remove_numbers_from_string(s) for s in eval(dom)]))].append([train_data, dev_data, test_data])

    train_data, valid_data, test_data = [], [], []
    table = []
    for dom, list_of_split in data_by_domain_new.items():
        train, valid, test = [], [], []
        for [tr,va,te] in list_of_split:
            train += rename_service_dialogue(tr,dom)
            valid += rename_service_dialogue(va,dom)
            test += rename_service_dialogue(te,dom)
        table.append({"dom":dom, "train":len(train), "valid":len(valid), "test":len(test)})
        train_data += train
        valid_data += valid
        test_data += test
    print(tabulate(table, headers="keys"))
    return train_data,valid_data,test_data


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

def remove_numbers_from_string(s):
    numbers = re.findall(r'_\d+', s)
    for n in numbers:
        s = s.replace(n,"")
    s = s.lower()
    if(s=="hotels"): s = "hotel"
    if(s=="restaurants"): s = "restaurant"
    if(s=="flights"): s = "flight"
    if(s=="movies"): s = "movie"
    return s.lower()

def get_dict(DST):
    di = defaultdict(lambda: defaultdict(str))
    for frame in DST:
        if(frame["state"]["active_intent"]!="NONE"):
            for k,v in frame["state"]['slot_values'].items():
                di[frame["state"]["active_intent"]][k] = v[0]
    return di


get_data_loaders(tokenizer)
