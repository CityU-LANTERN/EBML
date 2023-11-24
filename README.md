

## Secure Out-of-Distribution Task Generalization with Energy-Based Models
This repo contains a pytorch implementation for the paper: [Secure Out-of-Distribution Task Generalization with Energy-Based Models](https://openreview.net/pdf?id=tt7bQnTdRm) at Neurips 2023
### Setups
The code was tested with:
- python 3.8
- pytorch 1.12.1+cu116

Please see *requirements.txt* for other required packages.

The processed drugOOD dataset (~2GB) can be downloaded [here](https://drive.google.com/drive/folders/1Btva2-8NslXnpyivLUv2jtNbUS8Ba7l7?usp=sharing). You should save the data files to _src/datasets/drug/Ibap_general_ic50_size/_


### Experiments
For meta-training and meta-testing
> 
> `python run.py --exp {sinusoids, drug etc ...} --version ebml --seed 0 --mode train test --n_runs {number of runs}`

The full lists of hyperparameters used for the experiments can be found in the *config* directory. Your runs / training logs / testing results will all be saved in a directory called _experiments_ by default, you can change this by modifying the config .yaml file. 

You can also meta-test selected checkpoints by executing after meta-training.
> 
> `python run.py --exp {sinusoids, drug etc...} --version ebml --seed 0 --mode test --test_model_paths path/to/checkpoint_1 path/to/checkpint_2 etc...`
> 


### Citation
If you find this repository useful in your research, please consider citing the following paper:

    @inproceedings{
	    chen2023secure,
	    title={Secure Out-of-Distribution Task Generalization with Energy-Based Models},
	    author={Shengzhuang Chen and Long-Kai Huang and Jonathan Richard Schwarz and Yilun Du and Ying Wei},
	    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
	    year={2023},
	}


### Contact
If you have any question please feel free to **Contact** Shengzhuang Chen **Email**: szchen9-c [at] my [dot] cityu [dot] edu [dot] hk  