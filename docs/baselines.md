








## Full List of Implemented Baselines

### General Command Format
The `approach` argument consists of the following components `[approach_specification]` ,`[network]`  and `[approach]` . The `approach` argument should be in the format of `[network]_[approach_specification]_[approach]`, for example, if `[network]=bert_kim`, `[approach_specification]=hat` and `[approach]=ncl`, the `approach` should be `bert_kim_hat_ncl`.

The full list so far is as follows:

 
| Model Name | Approach Specification| Network | Approach |
|--|--| -- | -- |
| L2 | l2 | bert/bert_kim/bert adapter/w2v_as/w2v | ncl |
| A-GEM | a-gem | bert/bert_kim/bert adapter/w2v_as/w2v/cnn/mlp | ncl |
| [DER++](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) | derpp | bert/bert_kim/bert adapter/w2v_as/w2v/cnn/mlp | ncl |
| [KAN](https://www.cs.uic.edu/~liub/publications/ECML-PKDD-2020.pdf) | kan | bert_gru/w2v_as_gru/w2v_gru | ncl|
| [SRK](https://www.cs.uic.edu/~swang/papers/lv_shared_knowledge_sentiment.pdf) | srk | bert_gru/w2v_as_gru/w2v_gru | ncl|
| EWC | ewc | bert/bert_kim/bert_adapter/w2v_as/w2v/cnn/mlp | ncl |
| [HAL](https://arxiv.org/abs/2002.08165) | hal | bert/bert_kim/bert_adapter/w2v_as/w2v/cnn/mlp | ncl |
| [UCL](https://papers.nips.cc/paper/2019/hash/2c3ddf4bf13852db711dd1901fb517fa-Abstract.html) | ucl | bert/bert_kim/bert_adapter/w2v_as/w2v/cnn/mlp | ncl |
| [OWM](https://www.nature.com/articles/s42256-019-0080-x) | owm | bert/bert_kim/bert_adapter/w2v_as/w2v/cnn/mlp | ncl |
| [HAT](http://proceedings.mlr.press/v80/serra18a.html)| hat | bert/bert_kim/bert_adapter/w2v_as/w2v/cnn/mlp | ncl|
| [CAT](https://proceedings.neurips.cc/paper/2020/file/d7488039246a405baf6a7cbc3613a56f-Paper.pdf)| cat | bert/bert_kim/w2v_as/w2v/cnn/mlp| ncl|
| [B-CL](https://aclanthology.org/2021.naacl-main.378.pdf)| capsule_mask |bert_adapter | ncl|
| [CLASSIC](https://aclanthology.org/2021.emnlp-main.550/) | amix | bert_adapter_mask | ncl|



#### Network

 - **bert**: use BERT (fine-tuning) as backbone network
 - **bert_kim**: use BERT (frozen) as backbone netowok (a fronzen BERT + a CNN classificatio0n network on top)
 - **bert_gru**: use BERT (frozen) as backbone netowok (a fronzen BERT + a GRU classificatio0n network on top)
 - **bert_adapter**: use BERT (adapter) as backbone network
 - **w2v**: use W2V as backbone network (W2V embedding + a CNN classification network on top)
 - **w2v_as**: use W2V as backbone network (W2V embedding which includes both review and the aspect + a CNN classificatio0n network on top)
 - **w2v_gru**: use W2V as backbone network (W2V embedding + a GRU classification network on top)
 - **cnn**: use CNN as backbone network (for image dataset)
 - **mlp**: use MLP as backbone network (for image dataset)

#### Approach
 - **ncl**: Naive Continual Learning, meaning no special continual learning is applied by default. If `[approach_speicifciation]` is given, it becomes the corresponding baseline model.
 - **one**: a non-continual learning which trains each task independently
 - **mtl**: a non-continual learning which trains all task together via multi-task learning

	
### Example Commands 
The example commands are in format of `run_train_[network]_[approach_specification]_[approach].sh` in the `./commands` folder. You can run different baselines by simply replacing the `[approach_specification]` and `[network]`  and `[approach]`  to find the corresponding commands.

