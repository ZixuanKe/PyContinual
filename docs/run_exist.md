





  
  
  
  
  
  
  
  
  
  
  
  
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
```python
    python  run.py \  
		--bert_model 'bert-base-uncased' \  
		--backbone bert_adapter \  
		--baseline ctr \  
		--task asc \  
		--eval_batch_size 128 \  
		--train_batch_size 32 \  
		--scenario til_classification \
		--idrandom 0  \
		--use_predefine_args
```	
Above shows a typical command to run PyContinual. Some of the arguments are easy to understand, We further explain some PyContinual arguments:  
  - `idrandom`: which random sequence you want to use  
  - `use_predefine_args`: we have provided some  pre-defined arguments in `./load_base_args.py`. This argument will tell the program to use the pre-defined arguments
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
 - `baseline`: which baseline you want to run. See [baselines.md](https://github.com/ZixuanKe/PyContinual/blob/master/docs/baselines.md) for more
 - `backbone`: which backbone model you want to use. See [baselines.md](https://github.com/ZixuanKe/PyContinual/blob/master/docs/baselines.md) for more
### Example Commands   
  
#### Examples:  
  

  To run B-CL:    
```python
    python  run.py \  
		--bert_model 'bert-base-uncased' \   
		--backbone bert_adapter \  
		--baseline b-cl\  
		--task asc \  
		--eval_batch_size 128 \  
		--train_batch_size 32 \  
		--scenario til_classification \  
		--idrandom 0  \
		--use_predefine_args
```
 To run CLASSIC:    
```python
    python  run.py \  
		--bert_model 'bert-base-uncased' \   
		--backbone bert_adapter \  
		--baseline classic\  
		--task asc \  
		--eval_batch_size 128 \  
		--train_batch_size 32 \  
		--scenario dil_classification \  
		--idrandom 0  \
		--use_predefine_args
 ```
  To run CTR:    
```python
    python  run.py \  
		--bert_model 'bert-base-uncased' \   
		--backbone bert_adapter \  
		--baseline ctr \  
		--task asc \  
		--eval_batch_size 128 \  
		--train_batch_size 32 \  
		--scenario til_classification \  
		--idrandom 0  \
		--use_predefine_args
 ```
  
#### Examples Results:  
The example results will be saved in `./res/til_classification/asc/` and `./res/dil_classification/asc/`
