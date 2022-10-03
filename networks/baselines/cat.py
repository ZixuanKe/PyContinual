from tqdm.auto import tqdm
import torch
import math
import numpy as np
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoTokenizer,
    AutoConfig,
    RobertaTokenizer,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    SchedulerType,
    set_seed,
)
from torch import nn
from itertools import zip_longest
from utils import utils
import os


# CAT: (1) open similar; (2) an additional attention metric



class Similarity():
    def __init__(self):
        self.similarities = []

    def set_similarities(self,similarity):
        self.similarities.append(similarity)

    def fix_length(self):
        return len(self.similarities)

    def get_similarities(self):
        return self.similarities


    def check_t(self,t):
        if t < len([sum(x) for x in zip_longest(*self.similarities, fillvalue=0)]) and [sum(x) for x in zip_longest(*self.similarities, fillvalue=0)][t] > 0:
            return True

        elif np.count_nonzero(self.similarities[t]) > 0:
            return True

        elif t < len(self.similarities[-1]) and self.similarities[-1][t] == 1:
            return True

        return False




def compute(self,model,train_loader,dev_loader,accelerator):


    progressive_main_transfer_path = os.path.join(self.args.output_dir + '/../', 'progressive_main_transfer' + str(self.args.seed))
    progressive_similarity_transfer_path = os.path.join(self.args.output_dir + '/../', 'progressive_similarity_transfer' + str(self.args.seed))

    if os.path.exists(progressive_main_transfer_path):
        main_transfer = np.loadtxt(progressive_main_transfer_path)
        similarity_transfer = np.loadtxt(progressive_similarity_transfer_path)


    else:
        main_transfer = np.zeros((self.args.ntasks, self.args.ntasks), dtype=np.float32)
        similarity_transfer = np.zeros((self.args.ntasks, self.args.ntasks), dtype=np.float32)


    self.args.is_cat = True
    cur_task = self.args.ft_task


    for prev_task in range(self.args.ft_task+1):
        print('prev_task: ',prev_task)
        self.args.eval_t = prev_task
        model,metric = train_model(self,prev_task,model,train_loader,dev_loader,accelerator)
        results = self.eval(model, dev_loader, metric, accelerator, eval_t=self.args.ft_task)   # use main metric
        dev_main = utils.lookfor_main_metric(results, self.args)
        main_transfer[cur_task, prev_task] = dev_main

        self.args.is_reference = False
        self.args.is_transfer = False


    accelerator.wait_for_everyone()
    # TODO: why different GPU　has differnt similarity?
    if accelerator.is_main_process:
        print('Save at transfer_acc')
        np.savetxt(progressive_main_transfer_path,main_transfer,'%.4f',delimiter='\t')

    similarity = [0]
    if cur_task > 0:
        acc_list = main_transfer[cur_task][:cur_task] #t from 0
        print('acc_list: ',acc_list)

        similarity = [0 if (acc_list[acc_id] <= main_transfer[cur_task][cur_task]) else 1 for acc_id in range(len(acc_list))] # remove all acc < 0.5

        for source_task in range(len(similarity)):
            similarity_transfer[cur_task,source_task]=similarity[source_task]

    if accelerator.is_main_process:
        print('Save similarity_transfer at: ' + self.args.output_dir + '_similarity_transfer')
        np.savetxt(progressive_similarity_transfer_path,similarity_transfer,'%.4f',delimiter='\t')


    print('similarity: ',similarity)
    # TODO: Testing is still needed, this is a multinode function

    accelerator.unwrap_model(model).model.active_adapters = 'adapter'
    accelerator.unwrap_model(model).model.train_adapter('adapter')

    accelerator.unwrap_model(model).model.readouts = None #sansity
    accelerator.unwrap_model(model).teacher = None

    self.args.is_cat = False
    self.args.eval_t = cur_task

    return similarity



def train_model(self,prev_task,model,train_loader,dev_loader,accelerator):

    if prev_task == self.args.ft_task: # use but fixed use teacher adapter
        self.args.is_transfer = False
        self.args.is_reference = True
        accelerator.unwrap_model(model).teacher.active_adapters = 'reference_adapter'
        accelerator.unwrap_model(model).teacher.train_adapter('reference_adapter')
        classifier_lr = self.args.classifier_lr
    else:
        self.args.is_transfer = True
        self.args.is_reference = False
        accelerator.unwrap_model(model).model.active_adapters = 'adapter'
        accelerator.unwrap_model(model).model.train_adapter('adapter')
        classifier_lr = self.args.classifier_lr

    for _ in range(prev_task): #sanity
        accelerator.unwrap_model(model).teacher.readouts[_] = None
        accelerator.unwrap_model(model).model.readouts[_] = None

        # I fix the feature extractor inside bart_model.py


    no_decay = ["bias", "LayerNorm.weight"]
    special_lr = ['prompt', 'adapter', 'classifier']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in accelerator.unwrap_model(model).named_parameters() if
                       not any(nd in n for nd in no_decay) and p.requires_grad and not any(
                           nd in n for nd in special_lr)],
            "weight_decay": self.args.weight_decay,
            "lr": self.args.learning_rate
        },
        {
            "params": [p for n, p in accelerator.unwrap_model(model).named_parameters() if
                       any(nd in n for nd in no_decay) and p.requires_grad and not any(
                           nd in n for nd in special_lr)],
            "weight_decay": 0.0,
            "lr": self.args.learning_rate
        },
        {
            "params": [p for n, p in accelerator.unwrap_model(model).named_parameters() if
                       p.requires_grad and 'adapter' in n],
            "weight_decay": 0.0,
            "lr": self.args.adapter_lr,  # must use a higher lr
        },
        {
            "params": [p for n, p in accelerator.unwrap_model(model).named_parameters() if p.requires_grad and 'classifier' in n],
            "weight_decay": 0.0,
            "lr": classifier_lr,  # must use a higher lr
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    # Scheduler and math around the number of training steps.

    lr_scheduler = get_scheduler(
        name=self.args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=self.args.num_warmup_steps,
        num_training_steps=self.args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    metric = utils.load_my_metric(self.args)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
    max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch

    # Train!

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    best_model = utils.get_model(model)
    best_main = -np.inf
    patience = self.args.patient
    global_step = 0  # This will be used by CLMOE if we choose 'auto_encoder' as the route type.

    for epoch in range(starting_epoch, self.args.num_train_epochs):
        model.train()

        for step, batch in enumerate(train_loader):
            self.args.s = self.args.smax  # for reference, open half of it

            outputs = model(batch)

            loss = outputs.loss
            # We keep track of the loss at each epoch
            loss = loss / self.args.gradient_accumulation_steps
            accelerator.backward(loss)

            if accelerator.is_main_process and epoch < 1 and step < 1:
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        if self.args.is_transfer:
                            print('CAT Transfer n,p： ', n, p.size())
                        else:
                            print('CAT Reference n,p： ', n, p.size())

            if step % self.args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                global_step += 1
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                progress_bar.set_description(
                    'Train CAT Iter (Epoch=%3d,loss=%5.3f)' % ((epoch, loss.item())))  # show the loss, mean while

        #     break
        # break
        if completed_steps >= max_train_steps:
            break

        if (
                self.args.task_name in self.args.generation or self.args.task_name in self.args.ner_datasets) and epoch % 2 == 0:  # no need to test for everyone

            results = self.eval(model, dev_loader, metric, accelerator, eval_t=self.args.ft_task)

            dev_main = utils.lookfor_main_metric(results, self.args)

            if epoch < self.args.num_train_epochs and best_main < dev_main:  # data is too small, we need to at least run some epoch
                best_main = dev_main
                best_model = utils.get_model(model)
                if accelerator.is_main_process: print(
                    "*Epoch {}, dev rouge1 = {:.4f}".format(epoch, dev_main))
                patience = self.args.patient  # reset
            else:
                if accelerator.is_main_process: print(
                    "Epoch {}, dev rouge1 = {:.4f}".format(epoch, dev_main))
                patience -= 1
                if patience <= 0: break

    if (self.args.task_name in self.args.generation or self.args.task_name in self.args.ner_datasets):
        utils.set_model_(model, best_model)

    return model,metric