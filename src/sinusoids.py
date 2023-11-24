from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from src.datasets.toy_datasets import SineRegression
from src.utils import load_config_from_yaml,set_random_seed,ensure_path,save_config_to_yaml
from src.eval import analyze_regression_results
import torch
import torch.nn as nn
import numpy as np
import csv
import argparse
import matplotlib
import importlib
import logging
import os
import sys
matplotlib.use("Agg")


def train_model(configs,starting_epoch,logger,hps):
    model = move_to_device(get_model(configs),configs.device)
    best_val_mse = 10.0
    trainset = SineRegression(configs.dataset)
    train_loader = DataLoader(dataset=trainset, shuffle=True, batch_size=configs.train.meta_batch_size, drop_last=False)
    #################################
    '''start of training routine '''
    #################################
    epoch = starting_epoch
    while (epoch - starting_epoch) < configs.train.max_epoch:
        trainset.set_mode('train')
        aggressive = False
        #############################
        '''training over N epoch'''
        #############################
        training_history=[]
        for progress in tqdm(range(configs.train.save_freq),
                             desc='epoch : {} -> {}'.format(epoch, epoch + configs.train.save_freq - 1),
                             bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            # iterate over batches in the current epoch
            for i, batch in enumerate(train_loader, 1):
                train_batch_summary = model.meta_update(batch=batch)
                terms, values = train_batch_summary
                training_history.append(values)
            loss_discriptor = dict(zip(terms, values))
        epoch += configs.train.save_freq

        ################
        """validation"""
        ################
        trainset.set_mode('val_id')
        val_loader = DataLoader(dataset=trainset, shuffle=True, batch_size=100,
                                  drop_last=False)
        for i, batch in enumerate(val_loader, 1):
            support, query, task_info = batch
            x_s, y_s = [_.cuda().float().detach() for _ in support]
            x_q, y_q = [_.cuda().float().detach() for _ in query]
            val_mse = model.evaluate_mse(num_phi_samples=configs.model.num_phi_samples,
                                                     niter=configs.sgld.decoder.niter,
                                                     x_q =x_q,
                                                     x_s =x_s,
                                                     y_q = y_q,
                                                     y_s = y_s )
            val_mse = val_mse.mean()
            if val_mse <= best_val_mse:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch' : epoch,
                }, configs.exp_dir + '/model_history/best_validation.pth')
                best_val_mse = val_mse
            break
        ###############
        '''logging'''
        ###############
        training_history = np.stack(training_history)
        metric_mean = training_history.mean(axis=0)
        metric_std = training_history.std(axis=0)
        log_content = f"EPOCH : {epoch}"
        for term, mean, std in zip(terms, metric_mean, metric_std):
            log_content += f"\n{term} : {mean:.3f}+/-{std:.3f}"
        log_content += f"val mse : {val_mse.mean():.3f}"
        logger.info(log_content)
        #################
        '''save model '''
        #################
        if not hps:
            torch.save({
                'model_state_dict': model.state_dict(),
            }, configs.exp_dir + '/model_history/checkpoint_{}.pth'.format(epoch))


def test_model(configs, test_model_path):
    print(f'testing model : {test_model_path}')
    # reload model and dataset
    model_name =  test_model_path.split('/')[-1]
    model_name = model_name.split('.')[0]
    checkpoint = torch.load(test_model_path)
    state_dict = checkpoint['model_state_dict']
    model = move_to_device(get_model(configs),configs.device)
    to_remove = [k for k in state_dict.keys() if 'gp' in k]  # exclude the set of trained gp posteriors ID training tasks

    for k in to_remove:
        print(f'excluding keys in state_dict : {k}')
        state_dict.pop(k)

    model.load_state_dict(state_dict)
    testset = SineRegression(configs.dataset)
    # for MC average
    test_loader = DataLoader(dataset=testset, shuffle=False, batch_size=500, drop_last=False)
    file_flag = False
    sample_batch = next(iter(test_loader))
    try:
        model.fill_latent_buffer(sample_batch)
    except AttributeError:
        pass
    # recreate test_loader
    test_loader = DataLoader(dataset=testset, shuffle=False, batch_size=1, drop_last=False)
    # return
    for split,loader_mode in [('ID','test_id'), ('OOD','test_ood')]:
        print(f'{"testing on ID dataset" :=^50}')
        testset.set_mode(loader_mode)
        for i, batch in enumerate(tqdm(test_loader), 1):
            task_results= model.evaluate_batch(batch, num_phi_samples=configs.model.num_phi_samples, niter=configs.sgld.decoder.niter)
            task_results.update({'OOD':[split]})
            if file_flag:
                with open(configs.exp_dir + '/mse_{}.csv'.format(model_name), 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(zip(*list(task_results.values())))
            else:
                with open(configs.exp_dir + '/mse_{}.csv'.format(model_name), 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(task_results.keys()))
                    writer.writeheader()
                    writer = csv.writer(csvfile)
                    writer.writerows(zip(*list(task_results.values())))
                file_flag = True

    print(f'{"analyzing final results" :=^50}')
    return analyze_regression_results(filename=model_name, path=configs.exp_dir)

def get_model(configs):
    ebml_model_py = importlib.import_module("src.ebml_models.{}_sine".format(configs.exp))
    model_initializer = getattr(ebml_model_py, 'EBMMetaRegressor')
    return model_initializer(configs)

def move_to_device(model, device):
    for name, module in model.named_children():
        if 'f_base_net' not in name:
            module.to(device)
        else:
            print(f'Module : {name} is excluded in .to(device) operation')
    return model

def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def run_train_test(args):
    if 'train' in args.mode:
        set_random_seed(args.seed)
        DEFAULT_CONFIG_PATH = './config/{0}/{1}.yaml'.format(args.exp,args.version)
        # make exp name
        configs= load_config_from_yaml(DEFAULT_CONFIG_PATH)
        configs.exp_dir = "./experiments/{0}/{1}".format(args.exp,args.version)
        if configs.exp_name:
            configs.exp_dir += '_' + configs.exp_name.replace(
            '.', 'f')
        configs.exp_dir +=f"/run_{args.seed}"
        # ensure exp dir exists
        ensure_path(configs.exp_dir)
        ensure_path(configs.exp_dir + '/energy')
        ensure_path(configs.exp_dir + '/model_history')
        ensure_path(configs.exp_dir + '/auroc')
        ensure_path(configs.exp_dir + '/distribution')
        ensure_path(configs.exp_dir + '/debug')
        # save configuration file
        configs.device = 'cuda:0'
        configs.exp_seed = args.seed
        save_config_to_yaml(configs, configs.exp_dir)
        # start training
        print(torch.cuda.get_device_name(configs.device))
        logger = setup_logging('main', configs.exp_dir)
        train_model(configs,0,logger,hps=False)
    if 'test' in args.mode:
        if "train" in args.mode:
            args.test_model_paths = [f"{configs.exp_dir +'/model_history/best_validation.pth'}",
                                     f"{configs.exp_dir +'/model_history/checkpoint_4000.pth'}",
                                     f"{configs.exp_dir +'/model_history/checkpoint_3000.pth'}"]  # defautl checkpoints to test
        for test_model_path in args.test_model_paths:
            set_random_seed(args.seed,)
            exp_dir = '/'.join(test_model_path.split('/')[:-2])
            # reload config
            configs = load_config_from_yaml(exp_dir+'/exp_config.yaml')
            test_model(configs, test_model_path)




