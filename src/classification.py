import copy
import torch
import numpy as np
np.seterr(all="ignore")
import os
import time
import wandb
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange
from .utils import load_config_from_yaml,save_config_to_yaml,set_random_seed,get_log_files,ValidationAccuracies,print_and_log,ensure_path,save_all_py_files_in_src, config_to_printable
from .datasets.meta_dataset_reader import MetaDatasetReader
import importlib
from collections import OrderedDict
from ood_metrics import auroc,aupr,fpr_at_95_tpr
from tabulate import tabulate
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet TensorFlow warnings
tf.compat.v1.disable_eager_execution()

# Constants
PRINT_FREQUENCY = 1000
FIXED_TESTSET_ROOT =os.environ['FIXED_TESTSET_ROOT']

class Learner:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.checkpoint_dir = configs.exp_dir
        self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final = get_log_files(self.checkpoint_dir)
        # initialize model
        self.model = self.init_model(configs)
        if hasattr(configs,'resume_model_path'):
            resume_iter_str = ''.join([s for s in configs.resume_model_path.split('/')[-1] if s.isdigit()])
            self.start_iteration = int(resume_iter_str)
            checkpoint = torch.load(configs.resume_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if hasattr(self.model,'scheduler'):
                self.model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print('Resume training from checkpoint : {}'.format(self.start_iteration))
        else:
            self.start_iteration = 0
        # initialize dataset split
        self.train_set, self.val_set, self.test_set = self.init_data(configs.dataset.name)
        if configs.dataset.name in ['meta-dataset','domainnet']:
            self.dataloader  = MetaDatasetReader(self.configs.dataset.data_dir,
                                                 'train_test',
                                                 self.train_set, self.val_set,
                                                 self.test_set,
                                                 self.configs.dataset.max_way_train,
                                                 self.configs.dataset.max_way_test,
                                                 self.configs.dataset.max_support_train,
                                                 self.configs.dataset.max_support_test,
                                                 way=self.configs.dataset.num_ways,
                                                 shot=self.configs.dataset.num_support,
                                                 query_train=self.configs.dataset.num_query,
                                                 query_test=self.configs.dataset.num_query,
                                                 oneshot=self.configs.test.fixed_way_one_shot)
        else:
            self.dataloader = MetaDatasetReader(self.configs.dataset.data_dir,
                                                'train_test',
                                                self.train_set, self.val_set,
                                                self.test_set,
                                                self.configs.dataset.max_way_train,
                                                self.configs.dataset.max_way_test,
                                                self.configs.dataset.max_support_train,
                                                self.configs.dataset.max_support_test,
                                                way = self.configs.dataset.num_ways,
                                                shot = self.configs.dataset.num_support,
                                                query_train = self.configs.dataset.num_query,
                                                query_test = self.configs.dataset.num_query,
                                                oneshot=self.configs.test.fixed_way_one_shot)
        self.validation_accuracies = ValidationAccuracies(self.val_set)

    def init_model(self,configs):
        def move_to_device(model,device):
            for name, module in model.named_children():
                if 'f_net' not in name:
                    if hasattr(module, 'hyper_net'):
                        module.hyper_net.to(device)
                    else:
                        module.to(device)
                else:
                    print(f'Module : {name} is excluded in .to(device) operation')
            return model

        ebml_model_py = importlib.import_module("src.ebml_models.EBML_{}".format(configs.exp))
        model_initializer = getattr(ebml_model_py, 'EBMLMetaClassifier')
        return move_to_device(model_initializer(configs=configs), self.device)

    def save_model(self, path):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.model.optimizer.state_dict(),
                    'scheduler_state_dict ': self.model.scheduler.state_dict() if hasattr(self.model,'scheduler') else None,
                    'task_buffer': self.model.task_buffer if hasattr(self.model,'task_buffer') else None,
                    'latent_buffer': self.model.latent_buffer if hasattr(self.model, 'latent_buffer') else None
                    }, path)

    def init_data(self, dataset):
        if dataset == "meta-dataset":
            """Original meta-dataset split w/o ilsvrc_2012"""
            train_set = ['omniglot','dtd','aircraft', 'cu_birds','vgg_flower', 'fungi','quickdraw']
            validation_set = train_set + ['mscoco']
            test_set = train_set + ['mscoco', 'traffic_sign','cifar10','cifar100','mnist'] 
        else:
            raise ValueError

        return train_set, validation_set, test_set

    def validate(self,session):
        accuracy_dict = {}
        for item in self.val_set:
            accuracies = []
            for _ in trange(self.configs.train.num_val_tasks_per_dataset):
                task_dict = self.dataloader.get_validation_task(item,session)
                task_results = self.model.evaluate_batch(task_dict, self.configs.model.num_phi_samples,
                                                         return_ood_scores=False)
                accuracies.append(task_results['no_tta']['accuracy'])
            accuracies = np.array(accuracies).reshape(-1)
            accuracy = accuracies.mean() * 100.0
            confidence = (196.0 * accuracies.std()) / np.sqrt(len(accuracies))
            accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}
        return accuracy_dict

    def train(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:

            wandb.init(project=f"{self.configs.dataset.name}",
                       config=self.configs,
                       name=self.configs.exp_dir.split('/')[-1],
                       dir=self.configs.exp_dir)
            wandb.watch(self.model, log="all", log_freq=100)  # log every 100 tasks

            print_and_log(self.logfile,'training on datasets : {}'.format(self.train_set))
            # config mixed precision training
            training_history = []
            total_iterations = self.configs.train.num_tasks + self.start_iteration
            self.model.optimizer.zero_grad()
            tic = time.time()
            # training iteration
            for iteration in trange(self.start_iteration, total_iterations):
                task_dict = self.dataloader.get_train_task(session)
                task_loss, loss_discriptor, task_acc = self.model.meta_update(task_dict)
                task_loss = task_loss / self.configs.train.meta_batch_size
                task_loss.backward(retain_graph=False)
                # record batch metrics
                keys, values = loss_discriptor

                if (iteration+1)%5:
                    loss_discriptor = dict(zip(keys, values))
                    wandb.log(loss_discriptor,step=iteration)

                training_history.append([task_acc] + values)
                # gradient update once evergy meta_batch_size
                if ((iteration + 1) % self.configs.train.meta_batch_size == 0) or (iteration == (total_iterations - 1)):
                    self.model.optimizer.step()
                    self.model.optimizer.zero_grad()
                    if hasattr(self.model,'scheduler'):
                        if self.model.scheduler is not None:
                            self.model.scheduler.step()

                # print and log
                if (iteration + 1) % PRINT_FREQUENCY == 0:
                    time_eclapsed = time.time() - tic
                    training_history = np.stack(training_history)
                    metric_mean = np.nanmean(training_history,axis=0)
                    metric_std = np.nanstd(training_history,axis=0)
                    keys.insert(0, 'Acc')
                    log_content = ""
                    for key, mean, std in zip(keys, metric_mean, metric_std):
                        log_content += f"{key} : {mean:.3f}+/-{std:.3f} \n"
                    print_and_log(self.logfile, f"Task [{iteration + 1}/{total_iterations}], Time [{time_eclapsed:.4f}] :")
                    print_and_log(self.logfile, log_content)
                    training_history = []
                    tic = time.time()
                # save model
                if (iteration+1) % self.configs.train.save_freq == 0:
                    self.save_model(path=self.checkpoint_dir + '/latest_checkpoint{}.pt'.format(iteration+1))
                    print_and_log(self.logfile, f'saved latest checkpoint to : {self.checkpoint_dir}')
                # validation
                if ((iteration + 1) % self.configs.train.val_freq == 0) and (iteration + 1) != total_iterations:
                    accuracy_dict = self.validate(session)
                    self.validation_accuracies.print(self.logfile, accuracy_dict)
                    if self.validation_accuracies.is_better(accuracy_dict):
                        self.validation_accuracies.replace(accuracy_dict)
                        self.save_model(path=self.checkpoint_path_validation)
                        print_and_log(self.logfile, 'Best validation model was updated.')
            # final model
            self.save_model(path=self.checkpoint_path_final)
            self.logfile.close()

    def test(self, test_model_path):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            model_name = test_model_path.split('/')[-1]
            model_name = model_name.split('.')[0]
            # load model
            checkpoint = torch.load(test_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            suffix=''
            if self.configs.tta.num_steps:
                if self.configs.tta.update_bn:
                    suffix +='_ttaBN'
                elif self.configs.tta.update_bn:
                    suffix +='_ttaFiLM'
                elif self.configs.tta.update_tsa:
                    suffix += '_ttaTSA'
            if self.configs.test.log_dir:
                suffix+='_'+self.configs.test.log_dir
            prefix = '5way1shot' if self.configs.test.fixed_way_one_shot else 'varying'
            model_name = prefix + model_name + suffix
            # prepare log files
            test_results_dir = self.configs.exp_dir + f'/test_results/{model_name}'
            if not os.path.exists(test_results_dir):
                os.makedirs(test_results_dir)
            logfile = open(f'{test_results_dir}/test_log.txt', "a+", buffering=1)
            print_and_log(logfile, f'{f"":=^50}')
            print_and_log(logfile, f'{f"Testing model {model_name}" :=^50}')
            print_and_log(logfile, 'ID datasets : {}'.format(self.train_set))
            print_and_log(logfile, 'OOD datasets : {}'.format([dataset for dataset in self.test_set if dataset not in self.train_set]))
            print_and_log(logfile, config_to_printable(self.configs.tta))
            print_and_log(logfile, f'{f"":=^50}\n')
            self.classfication_and_ood_task_detection(test_results_dir,
                                                      fixed_way_one_shot = self.configs.test.fixed_way_one_shot,
                                                      session =session)
            logfile.close()

    def classfication_and_ood_task_detection(self,test_results_dir,fixed_way_one_shot,session):
        all_results = {}  
        all_task_info = {}
        for n, item in enumerate(self.test_set):
            all_results[item]=[]
            for i in trange(self.configs.test.num_tasks_per_dataset):
                task_dict = self.dataloader.get_test_task(item,session)
                task_results = self.model.evaluate_batch(task_dict, self.configs.model.num_phi_samples, return_ood_scores=True)

                # record task results
                task_info={}
                for tta_mode in task_results.keys():
                    task_results[tta_mode].update({'OOD': item})
                all_results[item].append(copy.deepcopy(task_results))

                # record task meta info
                for info_key in task_info.keys():
                    if all_task_info.get(info_key, None):
                        all_task_info[info_key].append(task_info[info_key])
                    else:
                        all_task_info[info_key] = [task_info[info_key]]

            # print accuracy
            dataset_accuracies= '{0:} : \n'.format(item)
            for tta_mode in task_results.keys():
                if task_results.get(tta_mode, None):
                    accuracies = [task_results[tta_mode]['accuracy'] for task_results in all_results[item]]
                    accuracies = np.array(accuracies).reshape(-1)
                    accuracy = accuracies.mean() * 100.0
                    confidence = (196.0 * accuracies.std()) / np.sqrt(len(accuracies))
                    dataset_accuracies += '{0} : {1:3.2f}+/-{2:2.2f} \n'.format(tta_mode, accuracy, confidence)
                else:
                    pass
            print(dataset_accuracies)

        log_str = ''
        processed_results = {}
        for n, (dataset, dataset_results) in enumerate(all_results.items()):
            for tta_mode in dataset_results[0].keys():
                if n == 0:
                    processed_results[tta_mode]={}
                for key in dataset_results[0][tta_mode].keys():
                    if n == 0:
                        processed_results[tta_mode][key]=[]
                    processed_results[tta_mode][key].extend([task_results[tta_mode][key] for task_results in dataset_results])

        results_df = pd.DataFrame.from_dict(processed_results['no_tta']) # convert to pandas.dataframe
        log_str += self.ood_metrics(results_df, self.train_set)
        print(log_str)
        return None
    
    def ood_metrics(self, tasks_df, id_dataset) -> str:
        print(f'ID datasets for OOD metrics : {id_dataset}')
        if not isinstance(id_dataset, list):
            id_dataset = [id_dataset]
        y_true = (~tasks_df['OOD'].isin(id_dataset)).to_numpy()
        if y_true.sum() == len(y_true):
            log_results = 'all results contains only tasks from OOD datasets\n'
            log_results += 'skipping auroc, aupr and fpr95 calculation\n'
        else:
            log_results=""
            results_table = OrderedDict({'Criteria':[],'AUROC':[],'AUPR':[], 'FPR95':[]})
            for ood_criteria in list(tasks_df.columns):
                if ood_criteria in ['OOD','accuracy'] or type(tasks_df[ood_criteria].iloc[0]) != np.float64:
                    print(f'skipping col : {ood_criteria} for analysis')
                    print(type(tasks_df[ood_criteria].iloc[0]))
                    continue
                y_score = tasks_df[ood_criteria].to_numpy()
                results_table['Criteria'].append(ood_criteria)
                results_table['AUROC'].append(auroc(y_score,y_true))
                results_table['AUPR'].append(aupr(y_score,y_true))
                results_table['FPR95'].append(fpr_at_95_tpr(y_score,y_true))
            log_results += "\n"
            log_results += tabulate(results_table, headers="keys",tablefmt="grid")
            log_results += "\n"
        return log_results



def run_meta_dataset_exp(args):
    print("entered job loop")
    if 'train' in args.mode:
        print("entered training loop")
        set_random_seed(args.seed)
        if not args.resume_model_path:
            DEFAULT_CONFIG_PATH = './config/{}/{}.yaml'.format(args.exp,args.version)
            # make exp name
            configs = load_config_from_yaml(DEFAULT_CONFIG_PATH)
            configs.exp_dir = "./experiments/{}/".format(args.exp)
            configs.exp_dir = configs.exp_dir + configs.exp + '_'
            if configs.model.prior_sn:
                configs.exp_dir += 'sn_'
            suffix = f'PSTEP{configs.sgld.prior.step}_PETA{configs.sgld.prior.eta}_PNITER{configs.sgld.prior.niter}'.replace('.', 'f')
            configs.exp_dir += suffix
            if configs.exp_name:
                configs.exp_dir += '_'+configs.exp_name
            # ensure exp dir exists
            ensure_path(configs.exp_dir)
            # save configuration file
            save_config_to_yaml(configs, configs.exp_dir)
            # save .py files at run time
            assert not os.path.exists(configs.exp_dir +'/code_timestamp'), f'dir : {configs.exp_dir +"/code_timestamp"} already exists'
            save_all_py_files_in_src(source_path='src',destination_path=configs.exp_dir +'/code_timestamp')
        else:
            assert os.path.exists(args.resume_model_path), 'resume model path does not exist'
            exp_dir = '/'.join(args.resume_model_path.split('/')[:-1])
            # reload config
            configs = load_config_from_yaml(exp_dir + '/exp_config.yaml')
            configs.resume_model_path = args.resume_model_path
        # start training
        configs.device = 'cuda:0'
        configs.training = True
        print('Using GPU : ', torch.cuda.get_device_name(configs.device))
        learner = Learner(configs)
        learner.train()
    elif 'test' in args.mode:
        for test_model_path in args.test_model_paths:
            set_random_seed(args.seed,deterministic=True)
            exp_dir = '/'.join(test_model_path.split('/')[:-1])
            configs = load_config_from_yaml(exp_dir + '/exp_config.yaml')
            configs['test']['fixed_way_one_shot'] = True #test in 1shot
            # start testing
            configs.device = 'cuda:0'
            configs.training = False
            print('Using GPU : ', torch.cuda.get_device_name(configs.device))
            learner = Learner(configs)
            learner.test(test_model_path=test_model_path)






