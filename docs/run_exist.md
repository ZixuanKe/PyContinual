







## Run the Existing Baselines
We have implemented **40+** baselines, you can simply run them using the corresponding scripts

### Setup

 - Firstly, download the availabele data in [Drive](https://drive.google.com/file/d/10hCJHDKYVw0tzSHk6YZrRMFsqNs57fzW/view?usp=sharing). Unzip it in the root folder. After doing that, your should see `./dat` and `./data` in the root folder
 -  Secondly, go to the `./data_prep`, find the needed task sequence. Move it out to the root folder

### Format
    `run_train_[network]_[approach_specification]_[approach].sh`
    [network]: bert/bert_kim/bert_gru/bert_adapter/w2v_as
    [approach_specification]: optional, e.g. cat, hat, ewc, agem...
    [approach]: ncl/one/mtl/
    [more options please refere to .sh files, run.py and config.py.]
 
 ### Network

 - **bert**: use BERT (fine-tuning) as backbone network
 - **bert_kim**: use BERT (frozen) as backbone netowok (a fronzen BERT + a CNN classificatio0n network on top)
 - **bert_gru**: use BERT (frozen) as backbone netowok (a fronzen BERT + a GRU classificatio0n network on top)
 - **bert_adapter**: use BERT (adapter) as backbone network
 - **w2v**: use W2V as backbone network (W2V embedding + a CNN classification network on top)
 - **w2v_as**: use W2V as backbone network (W2V embedding which includes both review and the aspect + a CNN classificatio0n network on top)
 - **w2v_gru**: use W2V as backbone network (W2V embedding + a GRU classification network on top)
 
### Approach
 - **ncl**: Naive Continual Learning, meaning no special continual learning is applied by default. If `[approach_speicifciation]` is given, it becomes the corresponding baseline model.
 - **one**: a non-continual learning which trains each task independently
 - **mtl**: a non-continual learning which trains all task together via multi-task learning

### Approach Specification
See [baselines.md](https://github.com/ZixuanKe/PyContinual/blob/master/docs/baselines.md)

### Commands Location

	`./commands/[d/t]il_classification/[dataset]/`
	[d/t]: indicate task incremental learning (TIL) or domain incremental (DIL)
	[dataset]: indicate your datasets (see ./dataloader to see the available datasets, e.g. asc, dsc, nli...)
	
   

 ### Examples:
 
 To run B-CL:  
`./commands/til_classification/asc/run_train_bert_adapter_capsule_mask_ncl.sh`   
         
 To run KAN:  
`./commands/til_classification/asc/run_train_bert_gru_kan_ncl.sh` 

 To run CAT:  
`./commands/til_classification/asc/run_train_bert_kim_cat_ncl.sh`     

 ### Results:
The example results will be saved in `./res/til_classification/asc/`
 
