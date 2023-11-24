from tqdm import tqdm
from src.datasets.drug.data import DrugDataLoader
from src.utils import *
import torch
import numpy as np
import matplotlib
from src.eval import analyze_regression_results
import importlib
import csv
matplotlib.use("Agg")

def get_model(configs):
    ebml_model_py = importlib.import_module("src.ebml_models.{}_drug".format(configs.exp))
    model_initializer = getattr(ebml_model_py, 'EBMMetaRegressor')
    return model_initializer(configs)


def move_to_device(model, device):
    for name, module in model.named_children():
        if 'f_base_net' not in name:
            module.to(device)
        else:
            print(f'Module : {name} is excluded in .to(device) operation')
    return model


def train_model(configs,starting_epoch,logger):
    dataloader = DrugDataLoader(configs)
    model = move_to_device(get_model(configs), configs.device)
    model.optimizer.zero_grad()
    best_val_mse = 10.0
    patience = configs.train.max_patience
    for epoch in range(configs.train.max_epoch):
        # train for one epoch
        training_history = []
        train_batches = dataloader.get_train_batches(n_batches=500)
        for step, task_batch in enumerate(pbar:=tqdm(train_batches)):
            #todo: parallelize a batch of task to make training faster
            for task in task_batch:
                x_s, y_s, x_q, y_q = task
                x_s, y_s, x_q, y_q = x_s.squeeze().float().to(configs.device), \
                                 y_s.squeeze().float().to(configs.device), \
                                 x_q.squeeze().float().to(configs.device), \
                                 y_q.squeeze().float().to(configs.device)
                y_s = y_s.unsqueeze(-1)
                y_q = y_q.unsqueeze(-1)
                task_loss , train_batch_summary, task_mse = model.meta_update(x_s,y_s,x_q,y_q)
                terms, values = train_batch_summary
                training_history.append(values)
                task_loss = task_loss/configs.train.meta_batch_size
                task_loss.backward()
                pbar.set_description(f"EPOCH : {epoch}, TASK MSE : {task_mse:.3f}")
            # meta-update
            model.optimizer.step()
            model.optimizer.zero_grad()
            if hasattr(model,'scheduler'):
                model.scheduler.step()
            if configs.train.debug:
                print(f"DEBUG: x_s : {x_s.shape}, x_q : {x_q.shape}")
                if step>=10:
                    break
        # save model
        if not (epoch+1)% configs.train.save_freq:
            torch.save({'model_state_dict': model.state_dict(),},
                       configs.exp_dir + '/model_history/checkpoint_{}.pth'.format(epoch))
        # validation
        if not (epoch+1)% configs.train.val_freq:
            val_tasks = dataloader.get_val_batches()
            val_mse=[]
            for step,task in enumerate(tqdm(val_tasks)):
                x_s, y_s, x_q, y_q = task
                x_s, y_s, x_q, y_q = x_s.squeeze().float().to(configs.device), \
                                     y_s.squeeze().float().to(configs.device), \
                                     x_q.squeeze().float().to(configs.device), \
                                     y_q.squeeze().float().to(configs.device)
                y_s = y_s.unsqueeze(-1)
                y_q = y_q.unsqueeze(-1)
                val_mse.append(model.evaluate_mse(0, configs.model.num_phi_samples,
                                                         x_s=x_s, y_s=y_s, x_q=x_q, y_q=y_q))
            val_mse = np.mean(val_mse)
            # logging
            training_history = np.stack(training_history)
            metric_mean = training_history.mean(axis=0)
            metric_std = training_history.std(axis=0)
            log_content = "\n"
            for term, mean, std in zip(terms, metric_mean, metric_std):
                log_content += f"{term} : {mean:.3f}+/-{std:.3f} \n "
            log_content += f"val mse : {val_mse:.3f} \n"
            logger.info(log_content)
            # early stopping
            if val_mse <= best_val_mse:
                patience = configs.train.max_patience
                best_val_mse = val_mse
                torch.save({'model_state_dict': model.state_dict(), },
                           configs.exp_dir + '/model_history/best_validation.pth')
            else:
                patience -= 1
            if patience < 0:
                print('early stopping trigged at {} epoch'.format(epoch))
                break
        if epoch ==1 and configs.train.debug:
            break

def test_model(configs, test_model_path):
    print(f'testing model : {test_model_path}')
    # reload model and dataset
    model_name =  test_model_path.split('/')[-1]
    model_name = model_name.split('.')[0]
    checkpoint = torch.load(test_model_path)
    model = move_to_device(get_model(configs),configs.device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    dataloader = DrugDataLoader(configs)
    file_flag = False
    for split in ['ID','OOD']:
        all_test_tasks = dataloader.get_test_tasks(split, n_tasks=500) ## a list of test tasks
        for step, task in enumerate(pbar:= tqdm(all_test_tasks)):
            pbar.set_description(f"Testing {split} tasks")
            x_s, y_s, x_q, y_q = task
            x_s, y_s, x_q, y_q = x_s.float().to(configs.device), \
                                 y_s.float().to(configs.device), \
                                 x_q.float().to(configs.device), \
                                 y_q.float().to(configs.device)
            assert x_s.dim() == 2
            y_s = y_s.unsqueeze(-1)
            y_q = y_q.unsqueeze(-1)
            task_results = model.evaluate_task(x_s,y_s,x_q,y_q,
                                                num_phi_samples=configs.model.num_phi_samples,
                                                niter=configs.sgld.decoder.niter)
            task_results.update({'OOD':split})
            if file_flag:
                with open(configs.exp_dir + '/mse_{}.csv'.format(model_name), 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(task_results.keys()))
                    writer.writerow(task_results)
            else:
                with open(configs.exp_dir + '/mse_{}.csv'.format(model_name), 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(task_results.keys()))
                    writer.writeheader()
                    writer.writerow(task_results)
                file_flag = True
    print(f'{"analyzing final results" :=^50}')
    return analyze_regression_results(filename=model_name, path=configs.exp_dir)

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


def run_drug_exp(args):
    set_random_seed(args.seed)
    if 'train' in args.mode:
        DEFAULT_CONFIG_PATH = './config/drug/{}.yaml'.format(args.version)
        # make exp name
        configs= load_config_from_yaml(DEFAULT_CONFIG_PATH)
        configs.exp_dir = "./experiments/{0}/{1}/{2}".format(args.exp, configs.dataset.name,args.version + '_')
        if configs.model.prior_sn: configs.exp_dir += 'sn_'
        if configs.loss.contrastive_loss: configs.exp_dir += 'contrast_'
        suffix = f'STEP{configs.sgld.prior.step}_ETA{configs.sgld.prior.eta}_NITER{configs.sgld.prior.niter}'.replace(
            '.', 'f')
        configs.exp_dir += suffix
        if configs.exp_name:
            configs.exp_dir += '_' + configs.exp_name
        # ensure exp dir exists
        ensure_path(configs.exp_dir)
        ensure_path(configs.exp_dir + '/energy')
        ensure_path(configs.exp_dir + '/model_history')
        ensure_path(configs.exp_dir + '/auroc')
        ensure_path(configs.exp_dir + '/distribution')
        ensure_path(configs.exp_dir + '/debug')
        # save configuration file
        configs.device = 'cuda:0'
        configs.seed = args.seed
        save_config_to_yaml(configs, configs.exp_dir)
        # start training
        print(torch.cuda.get_device_name(configs.device))
        logger = setup_logging('main', configs.exp_dir)
        train_model(configs,0,logger)
    if "test" in args.mode:
        if "train" in args.mode:
            args.test_model_paths = [f"{configs.exp_dir + '/model_history/checkpoint_99.pth'}"]  # defautl checkpoints to test
        for test_model_path in args.test_model_paths:
            # if args.resume:  # resume training
            #     path = args.resume_dir + '/model_history/checkpoint_{}.pth'.format(args.resume_epoch)
            # load config from test_model_path
            exp_dir = '/'.join(test_model_path.split('/')[:-2])
            # reload config
            configs = load_config_from_yaml(exp_dir+'/exp_config.yaml')
            test_model(configs, test_model_path)