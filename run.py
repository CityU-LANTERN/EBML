import copy
import os
import argparse
import importlib

if __name__ == '__main__':

    experiments={"sinusoids" : ['src.sinusoids','run_train_test'],
                 "multi-sinusoids":['src.sinusoids','run_train_test'],
                 "meta-dataset":['src.classification','run_meta_dataset_exp'],
                 "domainnet":['src.classification','run_meta_dataset_exp'],
                 "mini-imagenet":['src.classification','run_meta_dataset_exp'],
                 "drug": ['src.drug','run_drug_exp']}

    parser = argparse.ArgumentParser(description="Train and test model")
    parser.add_argument('--exp', type=str, choices=["drug","sinusoids", "multi-sinusoids", "meta-dataset","domainnet","mini-imagenet"])
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_runs',type=int,default=1, help="number of repeats to run")
    parser.add_argument('--mode', nargs='+', default=[], choices=["train","test"], help="Path to CSV for input to model training")
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--resume_model_path',type=str, default=None)
    parser.add_argument('--test_model_paths', nargs='+', default=[], help='Path to test model checkpoint')
    parser.add_argument('--wandb_online', dest='wandb_online', action='store_true', help='Path to test model checkpoint')
    parser.add_argument('--wandb_offline', dest='wandb_offline', action='store_true',
                        help='Path to test model checkpoint')
    args = parser.parse_args()

    os.environ['WANDB_MODE'] = 'disabled'
    if args.wandb_online:
        os.environ['WANDB_MODE'] = 'online'
    elif args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'
    routine = importlib.import_module(experiments[args.exp][0])
    job = getattr(routine,experiments[args.exp][1])
    if args.n_runs > 1:
        for seed in range(args.seed, args.seed+args.n_runs):
            args_copy = copy.deepcopy(args)
            args_copy.seed = seed
            job(args_copy)
    else:
        job(args)
