from tqdm.auto import tqdm
import torch
import torch.distributed as dist
import os


def gather_importance(head_importance):
    head_importance_list = [torch.zeros_like(head_importance) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=head_importance_list, tensor=head_importance.contiguous()) # everyone need to do this
    head_importance_list = torch.stack(head_importance_list)
    head_importance = torch.mean(head_importance_list,dim=0)
    return head_importance



def derpp_compute(train_dataloader_prune, model, buffer, args):


    buffer_path = os.path.join(args.output_dir, 'buffer')

    with torch.no_grad():
        total = args.buffer_size_per_dataset
        for iteration in range(total // args.per_device_train_pool_batch_size + 1):
            left = total - iteration * args.per_device_train_pool_batch_size
            inputs = next(iter(train_dataloader_prune))# only one batch

            if args.task_name in args.classification:
                outputs = model.model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['cls_labels'], output_hidden_states=True)
                buffer.add_data(
                    input_ids=inputs['input_ids'][:left],
                    labels=inputs['cls_labels'][:left], 
                    logits=outputs.hidden_states[-1][:left],
                    attention_mask=inputs['attention_mask'][:left],
                    task=inputs['task'][:left]
                )
            else:
                outputs = model.model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'], output_hidden_states=True)
                buffer.add_data(
                    input_ids=inputs['input_ids'][:left], 
                    labels=inputs['labels'][:left], 
                    logits=outputs.hidden_states[-1][:left], 
                    attention_mask=inputs['attention_mask'][:left],
                    task=inputs['task'][:left]
                )

    print('buffer size: ',buffer.get_size())
    torch.save(buffer, buffer_path)