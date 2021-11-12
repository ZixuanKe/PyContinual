

  
  
  
  
  
  
  
  
  
  
  
  
## Run the Existing Baselines  
We have implemented **40+** baselines, you can simply run them using the corresponding scripts  
  
### Setup  
  
 - Firstly, download the availabele data in [Drive](https://drive.google.com/file/d/10hCJHDKYVw0tzSHk6YZrRMFsqNs57fzW/view?usp=sharing). Unzip it in the root folder. After doing that, you should see `./dat` and `./data` in the root folder (If you are going to use W2V, you can download the W2V from [here](https://drive.google.com/file/d/1BFrnjV0LMfsnPcTcHyQ4LKHWV065GzBq/view) and put it in the root folder)  
 - Secondly, go to the `./data_seq`, find the needed task sequence (e.g. asc_random). Move it out to the root folder  
- Thirdly, install the required packages (alternatively, you may also refer to the `requirements.txt`  by running `conda create --name pycontinual --file requirements.txt`)
	- `conda create -n pycontinual python=3.7`  
	 - `conda activate pycontinual`  
	 - `conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch`  
	 - `pip install transformers==4.10.2`  
	 - `conda install datasets`
	 - Depends on baselines, you may be prompted to install more packages. You can simply install them by running `conda install <pkg>`
	  
### General Command Format  
     python run.py \   
       --task dsc\   
       --ntasks 10 \    
       --idrandom $id \    
       --scenario til_classification \     
       --approach bert_kim_hat_ncl \    
       --eval_batch_size 128 \    
       --train_batch_size 32 \    
       --num_train_epochs 10   
Above shows a typical command to run PyContinual. Some of the arguments are easy to understand, We further explain some PyContinual arguments:  
  
 - `task` There are 4 supported taskls so far  
   - `asc`: aspect ssentiment classification  
   - `dsc`: document sentiemnt classification  
   - `ssc`: sentiment classification  
   - `newsgroup`: 20 newsgroup  
   - `nli`: naural language inference  
   - `CelebA, CIFAR10, CIFAR100, FashionMNIST, F-EMNIST, MNIST, VLCS`: image datasets, as their names indicated  
 - `scenario` There are 2 supproted senario  
   - `til_classification`: task incremental learning for classification task  
   - `dil_classification`: domain incremental learning for classification task  
 - `appraoch`: This is to indicate the baseline, backbone and approach you want to run. For example, `bert_kim_hat_ncl` means you use `bert_kim` as backbone model, `hat` as your baseline and `ncl` as your approach. See [baselines.md](https://github.com/ZixuanKe/PyContinual/blob/master/docs/baselines.md) for more  
### Example Commands   
#### Example Commands Location  
  
 `./commands/[d/t]il_classification/[dataset]/` [d/t]: indicate task incremental learning (TIL) or domain incremental (DIL) [dataset]: indicate your datasets (see ./dataloader to see the available datasets, e.g. asc, dsc, nli...)  
  
#### Format  
 `run_train_[network]_[approach_specification]_[approach].sh` [network]: bert/bert_kim/bert_gru/bert_adapter/w2v_as [approach_specification]: optional, e.g. cat, hat, ewc, agem... [approach]: ncl/one/mtl/ [more options please refere to .sh files, run.py and config.py.]  For full list of implemented baselines, please see [baselines.md](https://github.com/ZixuanKe/PyContinual/blob/master/docs/baselines.md)  
  
     
  
#### Examples:  
  
  To run CLASSIC:    
`./commands/dil_classification/asc/run_train_bert_adapter_amix_ncl.sh`   
  To run B-CL:    
`./commands/til_classification/asc/run_train_bert_adapter_capsule_mask_ncl.sh`     
           
 To run KAN:    
`./commands/til_classification/asc/run_train_bert_gru_kan_ncl.sh`   
  To run CAT:    
`./commands/til_classification/asc/run_train_bert_kim_cat_ncl.sh`   
#### Examples Results:  
The example results will be saved in `./res/til_classification/asc/` and `./res/dil_classification/asc/`
