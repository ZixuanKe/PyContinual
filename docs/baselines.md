











## Full List of Implemented Baselines

We introduce all supported baselines and backbones.
   - `Paper`: the model name and its reference paper
   - `baseline`: the baseline name (argument `--baseline`)
	   - `ncl`: Naive Continual Learning, meaning no special continual learning is applied by default.
	   - `one`: Trains each task independently (ONE model per task)
	   - `mtl`: Trains all task together via Multi-Task Learning
   - `backbone`: the supported backbones for the corresponding baseline (argument `--backbone`)
	 - `bert`: use BERT (fine-tuning) as backbone network
	 - `bert_frozen`: use BERT (frozen) as backbone netowok (a fronzen BERT + a CNN/GRU classification network on top)
	 - `bert_adapter`: use BERT (adapter) as backbone network
	 - `w2v`: use W2V as backbone network (W2V embedding + a CNN classification network on top)
	 - `w2v_as (specific for Aspect Sentiment Classification)`: use W2V as backbone network (W2V embedding which includes both review and the aspect + a CNN classification network on top)
	 - `cnn`: use CNN as backbone network (for image dataset only)
	 - `mlp`: use MLP as backbone network (for image dataset only)
  - `task`: the supported task for the corresponding baseline (argument `--task`)
	  - `language` dataset includes `asc/dsc/ssc/nli/newsgroup'`
	  - `image` dataset includes `celeba/femnist/vlcs/cifar10/mnist/fashionmnist/cifar100` 




| Paper| Baseline| Backbone| task|
|--|--| -- | -- |
| NCL| ncl | bert/bert_frozen/bert_adapter/w2v_as/w2v | language, image|
| ONE | one | bert/bert_frozen/bert_adapter/w2v_as/w2v | language, image|
| MTL | mtl | bert/bert_frozen/bert_adapter/w2v_as/w2v | language, image|
| [L2](https://arxiv.org/abs/1612.00796) | l2 | bert/bert_frozen/bert_adapter/w2v_as/w2v | language, image|
| [A-GEM](https://arxiv.org/abs/1812.00420) | a-gem | bert/bert_frozen/bert_adapter/w2v_as/w2v/cnn/mlp | language, image|
| [DER++](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) | derpp | bert/bert_frozen/bert_adapter/w2v_as/w2v/cnn/mlp | language|
| [KAN](https://www.cs.uic.edu/~liub/publications/ECML-PKDD-2020.pdf) | kan | bert_frozen/w2v_as/w2v | language|
| [SRK](https://www.cs.uic.edu/~swang/papers/lv_shared_knowledge_sentiment.pdf) | srk | bert_frozen/w2v_as/w2v | language, image|
| [EWC](https://arxiv.org/abs/1612.00796) | ewc | bert/bert_frozen/bert_adapter/w2v_as/w2v/cnn/mlp |language, image|
| [HAL](https://arxiv.org/abs/2002.08165) | hal | bert/bert_frozen/bert_adapter/w2v_as/w2v/cnn/mlp | language, image|
| [UCL](https://papers.nips.cc/paper/2019/hash/2c3ddf4bf13852db711dd1901fb517fa-Abstract.html) | ucl | bert/bert_frozen/bert_adapter/w2v_as/w2v/cnn/mlp | language, image|
| [OWM](https://www.nature.com/articles/s42256-019-0080-x) | owm | bert/bert_frozen/bert_adapter/w2v_as/w2v/cnn/mlp | language, image|
| [ACL](https://arxiv.org/abs/2003.09553) | acl| cnn/mlp | image|
| [HAT](http://proceedings.mlr.press/v80/serra18a.html)| hat | bert/bert_frozen/bert_adapter/w2v_as/w2v/cnn/mlp | language, image|
| [CAT](https://proceedings.neurips.cc/paper/2020/file/d7488039246a405baf6a7cbc3613a56f-Paper.pdf)| cat | bert/bert_frozen/w2v_as/w2v/cnn/mlp| language, image|
| [B-CL](https://aclanthology.org/2021.naacl-main.378.pdf)| b-cl|bert_adapter | language|
| [CLASSIC](https://aclanthology.org/2021.emnlp-main.550/) | classic| bert_adapter | language|
| [CTR](https://proceedings.neurips.cc/paper/2021/hash/bcd0049c35799cdf57d06eaf2eb3cff6-Abstract.html) | ctr| bert_adapter | language|



