import logging
import math

import numpy as np
import os
import torch
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import shutil
import nltk
from approaches import after_finetune, before_finetune
from sklearn.metrics import f1_score
from utils import utils




class Appr(object):

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.config = config
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.mse = torch.nn.MSELoss()

        utils.lookfor_baseline_variable(self,args)



    def postprocess_text(self, preds, labels):

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = self.args.num_warmup_steps if self.args.num_warmup_steps > 0 else math.ceil(num_training_steps * self.args.warmup_ratio)
        return warmup_steps

    def train(self, model, train_loader, train_dataset, dev_loader, test_loaders, train_pool_loader, accelerator):

        # for n, p in model.named_parameters():
        #     if 'shared' in n:
        #         p.requires_grad = False


        # if accelerator.is_main_process:
        #     for n, p in model.named_parameters():
        #         if p.requires_grad:
        #             print('n,p： ', n, p.grad)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.

        no_decay = ["bias", "LayerNorm.weight"]
        special_lr = ['prompt', 'adapter', 'classifier']
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and p.requires_grad and not any(
                               nd in n for nd in special_lr)],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and p.requires_grad and not any(
                               nd in n for nd in special_lr)],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad and 'prompt' in n],
                "weight_decay": 0.0,
                "lr": self.args.prompt_lr,  # must use a higher lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           p.requires_grad and 'adapter' in n],
                "weight_decay": 0.0,
                "lr": self.args.adapter_lr,  # must use a higher lr
            },
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' in n],
                "weight_decay": 0.0,
                "lr": self.args.classifier_lr,  # must use a higher lr
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        if self.args.warmup_ratio:
            self.args.num_warmup_steps=self.get_warmup_steps(self.args.max_train_steps)

            print('self.args.num_warmup_steps: ',self.args.num_warmup_steps)

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )


        # Prepare everything with our `accelerator`.
        model, optimizer, train_loader, train_pool_loader, dev_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader,train_pool_loader,dev_loader,lr_scheduler)


        # before training ***********************************************************************************************

        self,model, train_loader, dev_loader, accelerator,metric,mask_pre,mask_back,self_fisher = before_finetune.prepare(self,model, train_loader, dev_loader, accelerator)

        # before training ***********************************************************************************************
        #
        # if accelerator.is_main_process:
        #     for n, p in model.named_parameters():
        #         if p.requires_grad:
        #             print('+ n,p： ', n, p.grad)
        # exit()


        if not self.args.eval_checkpoint:
            # We need to recalculate our total training steps as the size of the training dataloader may have changed.
            num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch

            # Train!
            total_batch_size = self.args.per_device_train_batch_size * accelerator.num_processes * self.args.gradient_accumulation_steps

            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {self.args.num_train_epochs}, Pre-trained Model = {self.args.model_name_or_path}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}, Lamb = {self.args.lamb}")
            logger.info(f"  Learning Rate = {self.args.learning_rate}, Prompt Learning Rate = {self.args.prompt_lr} Adapter Learning Rate = {self.args.adapter_lr}, Classifier Learning Rate = {self.args.classifier_lr}, Warmup Num = {self.args.num_warmup_steps}")
            logger.info(f"  Total optimization steps = {self.args.max_train_steps}, NTokens={self.args.n_tokens}")
            logger.info(f"  Seq ID = {self.args.idrandom}, Task id = {self.args.ft_task}, Task Name = {self.args.task_name}, Num task = {self.args.ntasks}")

            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(self.args.max_train_steps), disable=not accelerator.is_local_main_process)
            completed_steps = 0
            starting_epoch = 0
            # Potentially load in the weights and states from a previous save
            if self.args.resume_from_checkpoint:
                if self.args.resume_from_checkpoint is not None or self.args.resume_from_checkpoint != "":
                    accelerator.print(f"Resumed from checkpoint: {self.args.resume_from_checkpoint}")
                    accelerator.load_state(self.args.resume_from_checkpoint)
                    path = os.path.basename(self.args.resume_from_checkpoint)
                else:
                    # Get the most recent checkpoint
                    dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                    dirs.sort(key=os.path.getctime)
                    path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                # Extract `epoch_{i}` or `step_{i}`
                training_difference = os.path.splitext(path)[0]

                if "epoch" in training_difference:
                    starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                    resume_step = None
                else:
                    resume_step = int(training_difference.replace("step_", ""))
                    starting_epoch = resume_step // len(train_loader)
                    resume_step -= starting_epoch * len(train_loader)

            best_model = utils.get_model(model)
            best_main = -np.inf
            patience = self.args.patient
            global_step = 0  # This will be used by CLMOE if we choose 'auto_encoder' as the route type.


            if accelerator.is_main_process:
                tensorboard_file = os.path.join(self.args.output_dir, str(self.args.task_name) + '_log')
                print('tensorboard_file: ', tensorboard_file)

                if os.path.isdir(tensorboard_file):
                    shutil.rmtree(tensorboard_file)
                writer = utils.setup_writer(tensorboard_file)


                #TODO: remove old output
                # delete previous model
                for saved_output_dir in self.args.saved_output_dir[:-2 ]:  # we need -2 so that we can load model
                    if os.path.isdir(saved_output_dir):
                        # shutil.rmtree(saved_output_dir)
                        for item in os.listdir(saved_output_dir):
                            if (item.endswith(".bin") or item.endswith(".json")) and 'adapter' not in item and 'head' not in item:
                                os.remove(saved_output_dir + item)


            if not self.args.eval_only:
                try:
                    for epoch in range(starting_epoch, self.args.num_train_epochs):
                        model.train()

                        if self.args.with_tracking:
                            total_loss = 0
                        for step, batch in enumerate(train_loader):
                            self.args.s = (self.args.smax - 1 / self.args.smax) * step / len(
                                train_loader) + 1 / self.args.smax

                            # We need to skip steps until we reach the resumed step
                            if self.args.resume_from_checkpoint and epoch == starting_epoch:
                                if resume_step is not None and step < resume_step:
                                    completed_steps += 1
                                    continue

                            if 'ewc' in self.args.baseline:
                                outputs = model(batch, self_fisher=self_fisher)
                            elif 'adapter_hat' in self.args.baseline or 'adapter_cat' in self.args.baseline  \
                                    or 'adapter_bcl' in self.args.baseline \
                                    or 'adapter_ctr' in self.args.baseline \
                                    or 'adapter_classic' in self.args.baseline:
                                model_ori = accelerator.unwrap_model(model)
                                masks = utils.mask(model, accelerator, self.args)
                                outputs = model(batch, masks=masks, mask_pre=mask_pre)

                            else:
                                outputs = model(batch)

                            loss = outputs.loss

                            # replay here
                            if ('ldbr' in self.args.baseline or 'derpp' in self.args.baseline) \
                                and not (self.args.buffer is None or self.args.buffer.is_empty()) \
                                and step % self.args.replay_freq == 0:
                                
                                replay_batch = self.args.buffer.get_datadict(size=batch['input_ids'].shape[0])
                                if self.args.task_name in self.args.classification:
                                    replay_batch['cls_labels'] = replay_batch['labels']
                                replay_outputs = model(replay_batch)
                                
                                loss += replay_outputs.loss * self.args.replay_beta
                                if 'derpp' in self.args.baseline:
                                    loss += self.mse(replay_outputs.hidden_states[-1], replay_batch['logits']) * self.args.replay_alpha
                            
                            # We keep track of the loss at each epoch
                            if self.args.with_tracking:
                                total_loss += loss.detach().float()
                            loss = loss / self.args.gradient_accumulation_steps
                            accelerator.backward(loss)

                            if accelerator.is_main_process and epoch < 1 and step < 1:
                                for n,p in model.named_parameters():
                                    if p.grad is not None:
                                        print('n,p,： ',n,p.size())
                                    # elif p.requires_grad:
                                    #     print('-n,p,： ',n,p.size())



                            if self.args.ft_task > 0 and \
                                    ('adapter_hat' in self.args.baseline
                                     or 'adapter_cat' in self.args.baseline
                                     or 'adapter_bcl' in self.args.baseline
                                     or 'adapter_ctr' in self.args.baseline
                                     or 'adapter_classic' in self.args.baseline):
                                for n, p in model.named_parameters():
                                    if n in mask_back and p.grad is not None:
                                        # print('block n: ',mask_back[n])
                                        # block  n: module.model.model.decoder.layers.9.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.weight
                                        p.grad.data *= mask_back[n]
                                    elif n in self.tsv_para and p.grad is not None and 'hat' not in self.args.baseline:
                                        p.grad.data *= utils.get_view_for_tsv(n, model_ori, self.args)  # open for general
                                    # elif 'adapter_cat' in self.args.baseline: #TODO: to open the mask, utils.mask already consder it
                                    #     p.grad.data *= utils.get_similar_mask(self.args.similarity, model, accelerator, self.args)
                            
                            if 'agem' in self.args.baseline:
                                model_ori = accelerator.unwrap_model(model)
                                from networks.baselines import agem
                                if not (self.args.buffer is None or self.args.buffer.is_empty()):
                                    agem.store_grad(model_ori.model.parameters, self.args.grad_xy, self.args.grad_dims)
                                    model_ori.model.zero_grad()

                                    replay_batch = self.args.buffer.get_datadict(self.args.buffer_size_per_dataset)
                                    # TODO: consider data loader if needed for efficient
                                    replay_batch['cls_labels'] = replay_batch['labels']
                                    outputs = model_ori(replay_batch)
                                    accelerator.backward(outputs.loss) # also make it cannot deal with task with different #classes, as DREPP

                                    agem.store_grad(model_ori.model.parameters, self.args.grad_er, self.args.grad_dims)
                                    #
                                    dot_prod = torch.dot(self.args.grad_xy, self.args.grad_er)
                                    if dot_prod.item() < 0:
                                        g_tilde = agem.project(gxy=self.args.grad_xy, ger=self.args.grad_er)
                                        agem.overwrite_grad(model_ori.model.parameters, g_tilde, self.args.grad_dims)
                                    else:
                                        agem.overwrite_grad(model_ori.model.parameters, self.args.grad_xy, self.args.grad_dims)


                            if 'adapter_hat' in self.args.baseline   \
                                    or 'adapter_cat' in self.args.baseline \
                                    or 'adapter_bcl' in self.args.baseline \
                                    or 'adapter_ctr' in self.args.baseline \
                                    or 'adapter_classic' in self.args.baseline:
                                # Compensate embedding gradients
                                for n, p in model.named_parameters():
                                    if ('adapters.e' in n or  'model.e' in n) and p.grad is not None:
                                        num = torch.cosh(
                                            torch.clamp(self.args.s * p.data, -self.args.thres_cosh,
                                                        self.args.thres_cosh)) + 1
                                        den = torch.cosh(p.data) + 1
                                        p.grad.data *= self.args.smax / self.args.s * num / den


                            if step % self.args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                                optimizer.step()
                                global_step += 1
                                lr_scheduler.step()
                                optimizer.zero_grad()
                                progress_bar.update(1)
                                completed_steps += 1
                                progress_bar.set_description(
                                    'Train Iter (Epoch=%3d,loss=%5.3f)' % ((epoch, loss.item())))  # show the loss, mean while

                                if 'adapter_hat' in self.args.baseline   \
                                        or 'adapter_cat' in self.args.baseline \
                                        or 'adapter_bcl' in self.args.baseline \
                                        or 'adapter_ctr' in self.args.baseline \
                                        or 'adapter_classic' in self.args.baseline:
                                    # Constrain embeddings
                                    for n, p in model.named_parameters():
                                        if 'adapters.e' in n or 'model.e' in n:
                                            p.data = torch.clamp(p.data, -self.args.thres_emb, self.args.thres_emb)


                                if accelerator.is_main_process:
                                    utils.log_loss(writer, scalar_value=loss.item(), global_step=global_step)
                                    if outputs.sum_loss is not None: utils.log_loss(writer, loss_name=' summerization loss', scalar_value=outputs.sum_loss.item(), global_step=global_step)
                                    if outputs.contrast_loss is not None: utils.log_loss(writer, loss_name=' contrast loss', scalar_value=outputs.contrast_loss.item(), global_step=global_step)


                        #
                        #     break
                        # break
                        if completed_steps >= self.args.max_train_steps:
                            break

                        # test_loader = test_loaders[self.args.ft_task]
                        # test_loader = accelerator.prepare(test_loader)
                        # self.eval(model, test_loader, metric,accelerator)

                        if (self.args.task_name in self.args.generation or self.args.task_name in self.args.ner_datasets) and epoch % 2 == 0: # no need to test for everyone

                            if 'prompt' in self.args.baseline or 'l2p' in self.args.baseline:
                                results = utils.prompt_eval(self=self,
                                                            model=model,dataloader=dev_loader,metric=metric,
                                                            eval_t=self.args.ft_task, pred_file=None, target_file=None, accelerator=accelerator)
                            else:
                                results = self.eval(model, dev_loader, metric, accelerator,eval_t=self.args.ft_task)

                            dev_main = utils.lookfor_main_metric(results,self.args)

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

                except KeyboardInterrupt:  # even if contro-C, I still want to save model
                    return

                # after training ***********************************************************************************************


        self = after_finetune.compute(self, model, train_pool_loader, self_fisher, mask_pre, accelerator)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        for eval_t in range(self.args.ft_task + 1):
            self.args.ori_task_name = self.args.task_name
            self.args.eval_t = eval_t # for adapter hat and so

            self.args.task_name = self.args.all_tasks[eval_t] #self.args.task_name has chaned
            metric = utils.load_my_metric(self.args)
            unwrapped_model.model.args = self.args # updated
            print('self.args.task_name_eval: ',self.args.task_name)


            if ('one' in self.args.baseline) and eval_t != self.args.ft_task:
                continue  # for one, I only care about forward results

            if self.args.only_eval_current_task and eval_t != self.args.ft_task:
                continue

            pred_file = os.path.join(self.args.output_dir.replace(self.args.ori_task_name,self.args.all_tasks[eval_t]), self.args.all_tasks[eval_t]+str(self.args.ft_task) + '_pred')
            target_file = os.path.join(self.args.output_dir, self.args.all_tasks[eval_t] + '_target')

            os.makedirs(self.args.output_dir.replace(self.args.ori_task_name,self.args.all_tasks[eval_t]), exist_ok=True)
            if os.path.exists(pred_file) and accelerator.is_main_process:
                os.remove(pred_file)
            if os.path.exists(target_file) and accelerator.is_main_process:
                os.remove(target_file)

            accelerator.wait_for_everyone()


            test_loader = test_loaders[eval_t]
            test_loader = accelerator.prepare(test_loader)

            if 'prompt' in self.args.baseline or 'l2p' in self.args.baseline:
                results = utils.prompt_eval(self=self,
                                            model=model,dataloader=test_loader,
                                            metric=metric,  eval_t=eval_t, pred_file=pred_file, target_file=target_file, accelerator=accelerator)
            else:
                results = self.eval(model, test_loader, metric, accelerator, eval_t, pred_file, target_file)
            # micro_f1, macro_f1, accuracy, total_loss / total_num

            #TODO: separate bleu and F1 for different datasets

            if accelerator.is_main_process:
                utils.write_result(results,eval_t,self.args)

        return

        # after training ***********************************************************************************************



    def eval(self, model, dataloader, metric, accelerator, eval_t=None, pred_file=None, target_file=None,tune_model=None,infer_model=None):
        model.eval()
        if self.args.val_max_target_length is None:
            self.args.val_max_target_length = self.args.max_target_length

        gen_kwargs = {
            "max_length": self.args.val_max_target_length if self.args is not None else self.config.max_length,
            "num_beams": self.args.num_beams,
            "min_length": self.args.val_min_target_length,
            "no_repeat_ngram_size": self.args.no_repeat_ngram_size,
        }

        samples_seen = 0

        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)

        label_list = []
        prediction_list = []
        total_loss = 0
        total_num = 0
        ppl_sigmoid = 0
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                self.args.s = self.args.smax
                if  self.args.task_name in self.args.asc_datasets or self.args.task_name in self.args.five_large_datasets:
                    outputs = model(batch)
                    real_b = batch["input_ids"].size(0)
                    loss = outputs.loss
                    if isinstance(outputs.logits, list):
                        outputs.logits = torch.cat(outputs.logits)
                    outp = outputs.logits

                    if 'mtl' in self.args.baseline or 'comb' in self.args.baseline:
                        pred = []
                        for out in outp:
                            pred.append(out.max(1)[1])  # out has different size
                        pred = torch.stack(pred).squeeze()
                    elif 'ldbr' in self.args.baseline:
                        pred = torch.stack(outp[0]).squeeze().max(1)[1]
                    else:
                        pred = outp.max(1)[1]

                    predictions = accelerator.gather(pred)
                    references = accelerator.gather(batch['cls_labels'])

                    if accelerator.is_main_process and 'ner' not in self.args.sequence_file:
                        print('predictions: ',predictions.cpu().numpy().tolist())
                        print('references: ',references.cpu().numpy().tolist())

                    total_loss += loss.data.cpu().numpy().item() * real_b
                    total_num += real_b
                    label_list += references.cpu().numpy().tolist()
                    prediction_list += predictions.cpu().numpy().tolist()

                    progress_bar.update(1)
                    # break


                    if 'generative' in self.args.baseline:
                        ppl_sigmoid += self.sigmoid(outputs.ppl)


                elif self.args.task_name in self.args.ner_datasets:
                    outputs = model(batch)
                    if isinstance(outputs.logits, list):
                        outputs.logits = torch.cat(outputs.logits)
                    predictions = outputs.logits.argmax(dim=-1)
                    labels = batch["cls_labels"]
                    if not self.args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                    predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(dataloader) - 1:
                            predictions_gathered = predictions_gathered[: len(dataloader.dataset) - samples_seen]
                            labels_gathered = labels_gathered[: len(dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += labels_gathered.shape[0]

                    # unwrapped_model = accelerator.unwrap_model(model)
                    preds, refs = self.get_labels(predictions_gathered, labels_gathered, eval_t)

                    metric.add_batch(
                        predictions=preds,
                        references=refs,
                    )  # predictions and preferences are expected to be a nested list of labels, not label_ids


                else: # summerization

                    model(batch) # use for supsup
                    if 'prompt' in self.args.baseline or 'l2p' in self.args.baseline:
                        inputs_embeds,attention_mask = tune_model.get_prompt_extended_input_exclude_label(batch["input_ids"],batch["attention_mask"],batch["labels"])

                        generated_tokens = infer_model.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            **gen_kwargs)

                    else:
                        if self.args.is_reference:
                            generated_tokens = accelerator.unwrap_model(model).teacher.generate(
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                **gen_kwargs,
                            )

                        else:
                            generated_tokens = accelerator.unwrap_model(model).model.generate(
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                **gen_kwargs,
                            )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=self.args.tokenizer.pad_token_id
                    )
                    labels = batch["labels"]

                    if not self.args.pad_to_max_length:
                        # If we did not pad to max length, we need to pad the labels too
                        labels = accelerator.pad_across_processes(batch["labels"], dim=1,
                                                                  pad_index=self.args.tokenizer.pad_token_id)

                    generated_tokens, labels = accelerator.gather((generated_tokens, labels))  # gather is a must
                    generated_tokens = generated_tokens.cpu().numpy()
                    labels = labels.cpu().numpy()

                    if self.args.ignore_pad_token_for_loss:
                        # Replace -100 in the labels as we can't decode them.
                        labels = np.where(labels != -100, labels, self.args.tokenizer.pad_token_id)
                    if isinstance(generated_tokens, tuple):
                        generated_tokens = generated_tokens[0]
                    decoded_preds = self.args.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = self.args.tokenizer.batch_decode(labels, skip_special_tokens=True)



                    decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)


                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(dataloader) - 1:
                            decoded_preds = decoded_preds[: len(dataloader.dataset) - samples_seen]
                            decoded_labels = decoded_labels[: len(dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += len(decoded_labels)

                    if accelerator.is_main_process and pred_file is not None and target_file is not None:
                        with open(pred_file, 'a') as f_pred_file, open(target_file, 'a') as f_target_file:
                            for decoded_pred in decoded_preds:
                                f_pred_file.writelines(decoded_pred.replace('\n',' ') + '\n')

                            for decoded_label in decoded_labels:
                                f_target_file.writelines(decoded_label.replace('\n',' ') + '\n')

                    if self.args.task_name in self.args.dialogue_datasets:
                        decoded_labels = [[label] for label in decoded_labels]


                    metric.add_batch(
                        predictions=decoded_preds,
                        references=decoded_labels,
                    )

                    progress_bar.update(1)
                    progress_bar.set_description('ROUGE Computation')  # show the loss, mean while

                # break

        if  self.args.task_name in self.args.asc_datasets or self.args.task_name in self.args.five_large_datasets:
            micro_f1 = f1_score(label_list, prediction_list, average='micro')
            macro_f1 = f1_score(label_list, prediction_list, average='macro')
            accuracy = sum([float(label_list[i] == prediction_list[i]) for i in range(len(label_list))]) * 1.0 / len(
                prediction_list)

            if accelerator.is_local_main_process:
                print('macro_f1: ', macro_f1)
                if 'generative' in self.args.baseline and eval_t is not None:
                    np.savetxt(os.path.join(self.args.output_dir + '/../', 'ppl_sigmoid' + str(eval_t)),
                               ppl_sigmoid.cpu().numpy(), '%.4f', delimiter='\t')

            results = {'micro_f1':micro_f1, 'macro_f1':macro_f1, 'accuracy':accuracy,'loss':total_loss / total_num}


            return results

        elif self.args.task_name in self.args.dialogue_datasets:
            result = metric.compute()
            logger.info(result)
            return result

        elif self.args.task_name in self.args.ner_datasets:
            eval_metric = self.compute_metrics(metric)
            print('eval_metric: ',eval_metric)
            return eval_metric
            #F1 is as important as macro F1, so that we can compare macro f1 directly

        else:

            result = metric.compute(use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            result = {k: round(v, 4) for k, v in result.items()}
            logger.info(result)

            return result


    def get_labels(self,predictions, references,eval_t): #TODO: need to change to "task", if we do MTL
        # Transform predictions and references tensos to numpy arrays
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list_dict[self.args.task_name][p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [self.label_list_dict[self.args.task_name][l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def compute_metrics(self,metric):
        results = metric.compute()
        if self.args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
