import copy
import csv
import matplotlib
import pandas as pd
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pickle
import numpy as np
from tabulate import tabulate
from ood_metrics import auroc,aupr,fpr_at_95_tpr
from collections import OrderedDict
from src.utils import print_and_log
import os

def _compute_epoch_stats(tasks_df, filename):
    log_file = open(filename, "w+")
    dfs = dict(tuple(tasks_df.groupby('OOD')))
    for key, value in dfs.items():
        for attribute in list(value.columns):
            if attribute not in ['OOD']:
                confidence = (1.96 * value[attribute].std()) / np.sqrt(len(value[attribute]))
                print_and_log(log_file,f'{key}_{attribute} : MEAN : {np.nanmean(value[attribute]):.3f} +/- {confidence:.3f} MEDIAN : {np.nanmedian(value[attribute]):.3f}')
    print('\n')
    log_file.close()

def analyze_regression_results(filename,path):
    def _plot_auroc_pdf(tasks_df, filename):
        '''
        Assume OOD = positive
        plot the AUROC for energy threshold-based OOD detection result on ID test and OOD test datasets
        :param tasks_df:
        :return:
        '''
        log_file = open(filename, "w+")
        log_results = ''
        # the first two collumns are dataset and mse
        ood_options = ['OOD']
        y_true = tasks_df['OOD'].isin(ood_options).to_numpy()
        num_oods = y_true.sum()
        fpr95s={}
        criterias = [x for x in list(tasks_df.columns) if not any(k in x for k in ['OOD',"MSE"])] # exclude gt OOD task ID and predictions
        for ood_criteria in criterias:
            y_score = tasks_df[ood_criteria].to_numpy()
            mask = np.isnan(y_score)
            if mask.any():
                print(f'OOD_score in {ood_criteria} contains nan values')
            mask = ~mask
            fpr95s[ood_criteria] = fpr_at_95_tpr(y_score[mask], y_true[mask])
            log_results += f'{f"{ood_criteria}":<15} ： AUROC : {auroc(y_score[mask], y_true[mask]):.4f}, AUPR: ' \
                           f'{aupr(y_score[mask], y_true[mask]): .4f}, FPR95: {fpr95s[ood_criteria]: .4f}\n'
        print_and_log(log_file,log_results)
        log_file.close()
        if fpr95s.get('data+kl+prior') and fpr95s['data+kl+prior'] <fpr95s['data'] :
            return fpr95s['data+kl']
        else :
            return 1.0

    def _plot_tasks_distribution_pdf(tasks_df, savepath):
        # features has header: ['ID', 'Epoch', 'f(x)', 'MSE', 'Supp E', 'OOD']
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        dfs = dict(tuple(tasks_df.groupby('OOD')))
        ID_TOKENS = ['ID']
        criterias = [x for x in list(tasks_df.columns) if x not in ['OOD']]
        for ood_criteria in criterias:
            fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            e_min = tasks_df[ood_criteria].min()
            e_max = tasks_df[ood_criteria].max()
            y = [10, 20]
            c = ['green', 'red']
            for key, value in dfs.items():
                if key in ID_TOKENS:
                    isOOD = 0
                else:
                    isOOD = 1
                a0.hist(value[ood_criteria], bins=100, alpha=0.5, range=(e_min, e_max), label=key)
                a1.scatter(value[ood_criteria], y[isOOD] * np.ones(value[ood_criteria].shape), marker='|',
                           color=c[isOOD], alpha=0.1)
            a0.legend()
            a0.set_xlabel(f'OOD score : {ood_criteria}')
            a0.set_ylabel('Frequency')
            a0.grid()
            a0.set_ylim(0, 50)
            a1.set_ylim(0, 50)
            a1.set_yticklabels([])
            plt.savefig(savepath +f'/{ood_criteria}.pdf', dpi=50)
            plt.close()

    results_df = pd.read_csv(path + '/mse_{}.csv'.format(filename), sep=',', header=0)
    _compute_epoch_stats(results_df, path +'/test_{}.txt'.format(filename) )
    fpr95 = _plot_auroc_pdf(results_df, path +'/auroc/{}.txt'.format(filename))
    _plot_tasks_distribution_pdf(results_df, path +'/distribution/{}'.format(filename))
    return fpr95

def analyze_classification_results(test_results_dir, raw_test_results, raw_phi_results, ID_datasets, hps=False):
    def _plot_auroc_pdf(tasks_df, id_dataset) -> str:
        '''
        Assume OOD = positive
        plot the AUROC for energy threshold-based OOD detection result on ID test and OOD test datasets
        :param tasks_df:
        :return:
        '''
        print(f'ID datasets for OOD metrics : {id_dataset}')
        if not isinstance(id_dataset, list):
            id_dataset = [id_dataset]
        #
        # y_label = tasks_df['OOD'].isin(id_dataset).int()
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

    def _plot_tasks_distribution_pdf(tasks_df, savepath):
        fig_path1 = savepath + f'/ood_scores_distribution'
        if not os.path.exists(fig_path1):
            os.makedirs(fig_path1)
        # features has header: ['ID', 'Epoch', 'f(x)', 'MSE', 'Supp E', 'OOD']
        dfs = dict(tuple(tasks_df.groupby('OOD')))
        for ood_criteria in list(tasks_df.columns):
            if ood_criteria in ['OOD', 'accuracy'] or type(tasks_df[ood_criteria].iloc[0]) != np.float64:
                continue
            fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
            e_min = tasks_df[ood_criteria].min()
            e_max = tasks_df[ood_criteria].max()
            y = [10, 20]
            c = ['green', 'red']
            for key, value in dfs.items():
                if key in ID_datasets:
                    isOOD = 0
                else:
                    isOOD = 1
                a0.hist(value[ood_criteria], bins=100, alpha=0.5, range=(e_min, e_max), label=key)
                a1.scatter(value[ood_criteria], y[isOOD] * np.ones(value[ood_criteria].shape), marker='|',
                           color=c[isOOD], alpha=0.1)
            a0.legend()
            a0.set_xlabel(f'OOD score : {ood_criteria}')
            a0.set_ylabel('Frequency')
            a0.grid()
            a0.set_ylim(0, 50)
            a1.set_ylim(0, 50)
            a1.set_yticklabels([])
            plt.savefig(fig_path1 + '/{}.pdf'.format(ood_criteria), dpi=50)
            plt.close()

    def _plot_ece_pdf(tasks_df, savepath, id_dataset):

        fig_path1 = savepath + f'/correlation_plots'
        if not os.path.exists(fig_path1):
            os.makedirs(fig_path1)

        dfs = dict(tuple(tasks_df.groupby('OOD')))

        for ood_criteria in list(tasks_df.columns):
            if ood_criteria in ['OOD', 'accuracy'] or type(tasks_df[ood_criteria].iloc[0]) != np.float64:
                continue
            # ece for each dataset
            n_bins = 10
            c = ['green', 'red']
            fig1, ece_axs = plt.subplots(3, 5, sharex=True, sharey=True, figsize=(20, 12))
            for plot_i, (key, value) in enumerate(dfs.items()):
                scores = value[ood_criteria].to_numpy()
                accuracies = value['accuracy'].to_numpy()
                e_min = scores.min()
                e_max = scores.max()
                scores = (scores - e_min) / (e_max - e_min)  * 2.0 -1.0 # normalized to  [0,1]
                accuracies = (accuracies-accuracies.min())/(accuracies.max()-accuracies.min()) * 2.0 -1.0
                isOOD = key not in id_dataset
                bin_weight,bin_acc,bin_conf =[],[],[]
                axs_x, axs_y = plot_i%5, plot_i//5
                for edge_start in np.arange(-1,1,2/n_bins):
                    sample_idx = np.less_equal(scores, edge_start + 2/n_bins)
                    bin_weight.append(sample_idx.sum()/len(sample_idx))
                    bin_acc.append(accuracies[sample_idx].mean())
                    bin_conf.append(scores[sample_idx].mean())
                ece = np.sum(np.array(bin_weight) * np.abs(np.array(bin_acc)-np.array(bin_conf)))
                rho = np.corrcoef(scores, accuracies)[0,1]
                ece_axs[axs_y, axs_x].stairs(bin_acc, np.arange(-1, 1 + 1 / n_bins, 2 / n_bins), fill=True, color=c[isOOD])
                ece_axs[axs_y, axs_x].set_title(f'{key} : {ece:.3f}/{rho:.3f}')
                ece_axs[axs_y, axs_x].set_xlabel(f'{ood_criteria}')
                ece_axs[axs_y, axs_x].set_ylabel('accuracy')
                ece_axs[axs_y, axs_x].grid()
                ece_axs[axs_y, axs_x].set_ylim(-1, 1)
                ece_axs[axs_y, axs_x].set_xlim(-1, 1)

            # ece for ID overall
            scores = tasks_df.loc[tasks_df['OOD'].isin(id_dataset)][ood_criteria].to_numpy()
            accuracies = tasks_df.loc[tasks_df['OOD'].isin(id_dataset)]['accuracy'].to_numpy()
            if len(scores):
                e_min = scores.min()
                e_max = scores.max()
                scores = (scores - e_min) / (e_max - e_min) * 2.0 - 1.0  # normalized to  [0,1]
                accuracies = (accuracies - accuracies.min()) / (accuracies.max() - accuracies.min()) * 2.0 - 1.0
                bin_weight, bin_acc, bin_conf = [], [], []
                axs_x, axs_y = 2, 2
                for edge_start in np.arange(-1, 1, 2 / n_bins):
                    sample_idx = np.less_equal(scores, edge_start + 2 / n_bins)
                    bin_weight.append(sample_idx.sum() / len(sample_idx))
                    bin_acc.append(accuracies[sample_idx].mean())
                    bin_conf.append(scores[sample_idx].mean())
                ece = np.sum(np.array(bin_weight) * np.abs(np.array(bin_acc) - np.array(bin_conf)))
                rho = np.corrcoef(scores, accuracies)[0, 1]
                ece_axs[axs_y, axs_x].stairs(bin_acc, np.arange(-1, 1 + 1 / n_bins, 2 / n_bins), fill=True, color='green')
                ece_axs[axs_y, axs_x].set_title(f'ID : {ece:.3f}/{rho:.3f}')
                ece_axs[axs_y, axs_x].set_xlabel(f'{ood_criteria}')
                ece_axs[axs_y, axs_x].set_ylabel('accuracy')
                ece_axs[axs_y, axs_x].grid()
                ece_axs[axs_y, axs_x].set_ylim(-1, 1)
                ece_axs[axs_y, axs_x].set_xlim(-1, 1)

            # ece for OOD overall
            scores = tasks_df.loc[~tasks_df['OOD'].isin(id_dataset)][ood_criteria].to_numpy()
            accuracies = tasks_df.loc[~tasks_df['OOD'].isin(id_dataset)]['accuracy'].to_numpy()
            if len(scores):
                e_min = scores.min()
                e_max = scores.max()
                scores = (scores - e_min) / (e_max - e_min) * 2.0 - 1.0  # normalized to  [0,1]
                accuracies = (accuracies - accuracies.min()) / (accuracies.max() - accuracies.min()) * 2.0 - 1.0
                bin_weight, bin_acc, bin_conf = [], [], []
                axs_x, axs_y = 3, 2
                for edge_start in np.arange(-1, 1, 2 / n_bins):
                    sample_idx = np.less_equal(scores, edge_start + 2 / n_bins)
                    bin_weight.append(sample_idx.sum() / len(sample_idx))
                    bin_acc.append(accuracies[sample_idx].mean())
                    bin_conf.append(scores[sample_idx].mean())
                ece = np.sum(np.array(bin_weight) * np.abs(np.array(bin_acc) - np.array(bin_conf)))
                rho = np.corrcoef(scores, accuracies)[0, 1]
                ece_axs[axs_y, axs_x].stairs(bin_acc, np.arange(-1, 1 + 1 / n_bins, 2 / n_bins), fill=True, color='red')
                ece_axs[axs_y, axs_x].set_title(f'OOD :  {ece:.3f}/{rho:.3f}')
                ece_axs[axs_y, axs_x].set_xlabel(f'{ood_criteria}')
                ece_axs[axs_y, axs_x].set_ylabel('accuracy')
                ece_axs[axs_y, axs_x].grid()
                ece_axs[axs_y, axs_x].set_ylim(-1, 1)
                ece_axs[axs_y, axs_x].set_xlim(-1, 1)

            # ece overall
            scores = tasks_df[ood_criteria].to_numpy()
            accuracies = tasks_df['accuracy'].to_numpy()
            e_min = scores.min()
            e_max = scores.max()
            scores = (scores - e_min) / (e_max - e_min) * 2.0 - 1.0  # normalized to  [0,1]
            accuracies = (accuracies - accuracies.min()) / (accuracies.max() - accuracies.min()) * 2.0 - 1.0
            bin_weight, bin_acc, bin_conf = [], [], []
            axs_x, axs_y = 4, 2
            for edge_start in np.arange(-1, 1, 2 / n_bins):
                sample_idx = np.less_equal(scores, edge_start + 2 / n_bins)
                bin_weight.append(sample_idx.sum() / len(sample_idx))
                bin_acc.append(accuracies[sample_idx].mean())
                bin_conf.append(scores[sample_idx].mean())
            ece = np.sum(np.array(bin_weight) * np.abs(np.array(bin_acc) - np.array(bin_conf)))
            rho = np.corrcoef(scores, accuracies)[0, 1]
            ece_axs[axs_y, axs_x].stairs(bin_acc, np.arange(-1, 1 + 1 / n_bins, 2 / n_bins), fill=True, color='blue')
            ece_axs[axs_y, axs_x].set_title(f'All :  {ece:.3f}/{rho:.3f}')
            ece_axs[axs_y, axs_x].set_xlabel(f'{ood_criteria}')
            ece_axs[axs_y, axs_x].set_ylabel('accuracy')
            ece_axs[axs_y, axs_x].grid()
            ece_axs[axs_y, axs_x].set_ylim(-1, 1)
            ece_axs[axs_y, axs_x].set_xlim(-1, 1)
            plt.savefig(fig_path1 + '/{}.pdf'.format(ood_criteria), dpi=50)
            plt.close()

    def _plot_correlation_pdf(tasks_df, savepath, id_dataset):

        fig_path2 = savepath + f'/energy_histograms'
        if not os.path.exists(fig_path2):
            os.makedirs(fig_path2)

        dfs = dict(tuple(tasks_df.groupby('OOD')))

        for ood_criteria in list(tasks_df.columns):
            if ood_criteria in ['OOD', 'accuracy'] or type(tasks_df[ood_criteria].iloc[0]) != np.float64:
                continue
            # ece for each dataset
            n_bins = 10
            c = ['green', 'red']
            fig1, his_axs = plt.subplots(3, 5, figsize=(20, 12))
            for plot_i, (key, value) in enumerate(dfs.items()):
                scores = value[ood_criteria].to_numpy()
                isOOD = key not in id_dataset
                counts,edges = np.histogram(scores)
                axs_x, axs_y = plot_i % 5, plot_i // 5
                his_axs[axs_y, axs_x].stairs(counts, edges, fill=True,color=c[isOOD])
                his_axs[axs_y, axs_x].set_title(f'{key}')
                his_axs[axs_y, axs_x].set_xlabel(f'{ood_criteria}')
                his_axs[axs_y, axs_x].set_ylabel('count')
                his_axs[axs_y, axs_x].grid()


            # ece for ID overall
            scores = tasks_df.loc[tasks_df['OOD'].isin(id_dataset)][ood_criteria].to_numpy()
            if len(scores):
                axs_x, axs_y = 2, 2
                counts, edges = np.histogram(scores)
                his_axs[axs_y, axs_x].stairs(counts, edges, fill=True, color='green')
                his_axs[axs_y, axs_x].set_title(f'ID')
                his_axs[axs_y, axs_x].set_xlabel(f'{ood_criteria}')
                his_axs[axs_y, axs_x].set_ylabel('count')
                his_axs[axs_y, axs_x].grid()

            # ece for OOD overall
            scores = tasks_df.loc[~tasks_df['OOD'].isin(id_dataset)][ood_criteria].to_numpy()
            if len(scores):
                axs_x, axs_y = 3, 2
                counts, edges = np.histogram(scores)
                his_axs[axs_y, axs_x].stairs(counts, edges, fill=True, color='red')
                his_axs[axs_y, axs_x].set_title(f'OOD')
                his_axs[axs_y, axs_x].set_xlabel(f'{ood_criteria}')
                his_axs[axs_y, axs_x].set_ylabel('count')
                his_axs[axs_y, axs_x].grid()

            plt.savefig(fig_path2 + '/{}.pdf'.format(ood_criteria), dpi=50)
            plt.close()

    def _plot_tta_trajectory_pdf(tta_results, savepath):
        tta_results_df = pd.DataFrame.from_dict(tta_results)
        dfs = dict(tuple(tta_results_df.groupby('OOD')))

        fig_path2 = savepath + f'/tta_trajectories'
        if not os.path.exists(fig_path2):
            os.makedirs(fig_path2)

        for metric,trajectory in tta_results.items():
            if metric not in ['OOD'] and type(trajectory[0])==list and len(trajectory[0]) > 1:
                fig = plt.figure(figsize=(8, 6))
                for dataset, values in dfs.items():
                    n_tasks = len(values[metric])
                    col_np = np.array(values[metric].tolist()).reshape(n_tasks, -1)
                    e_mean_item = np.mean(col_np ,0)
                    plt.plot(e_mean_item,'--v',label=dataset)
                plt.legend()
                plt.title(f'{metric}')
                plt.show()
                fig.savefig(fig_path2 + '/{}.pdf'.format(f'{metric}'), dpi=50)
                plt.close(fig)

    def _plot_tta_deltaacc_pdf(mode_results, base_results,savepath) -> str:
        fig_path2 = savepath + f'/tta_vs_delta_accuracy'
        if not os.path.exists(fig_path2):
            os.makedirs(fig_path2)

        log_results= ""
        results_table=OrderedDict({})
        for trajectory in mode_results.keys():
            if trajectory not in ['OOD'] and type(mode_results[trajectory][0])==list and len(mode_results[trajectory][0]) > 1:
                results_table.update({trajectory:[]})
                # plot delta energy vs delta accuray
                delta_acc = np.array(mode_results['accuracy']) - np.array(base_results['accuracy'])
                energy_trajectory = np.array(mode_results[trajectory])
                delta_energy = energy_trajectory[:, -1] - energy_trajectory[:, 0]  # end -start
                delta_energy = -delta_energy
                fig1, axs = plt.subplots(3, 5, figsize=(20, 12))
                for plot_i, dataset in enumerate(np.unique(mode_results['OOD'])):
                    idx = np.isin(mode_results['OOD'], [dataset])
                    axs_x, axs_y = plot_i % 5, plot_i // 5
                    rho = np.corrcoef(delta_energy[idx], delta_acc[idx])[0, 1]
                    axs[axs_y, axs_x].scatter(delta_energy[idx], delta_acc[idx], alpha=0.6, label=dataset)
                    axs[axs_y, axs_x].set_title(f'{dataset} : rho = {rho:.3f}')
                    axs[axs_y, axs_x].set_xlabel(r'-$\Delta$'+f'{trajectory.split("_")[0]}')
                    axs[axs_y, axs_x].set_ylabel(r'$\Delta$ accuracy')
                    axs[axs_y, axs_x].legend()
                    axs[axs_y, axs_x].grid()
                    results_table[trajectory].append(rho)
                if not results_table.get('dataset'):
                    results_table.update({'dataset' : np.unique(mode_results['OOD']).tolist()})
                plt.show()
                plt.savefig(fig_path2 + '/{}.pdf'.format(f'{trajectory.split("_")[0]}_vs_accuracy'), dpi=50)
                plt.close()
        results_table.move_to_end('dataset',last=False)
        log_results += tabulate(results_table, headers="keys",tablefmt="grid")
        log_results +='\n'
        return log_results

   # load pickle file
   #  file = open(f'{test_results_dir}/test_results.pickle', 'rb')
   #  raw_results = pickle.load(file)
   #  file.close()

    log_str = ''
    all_results = {}
    # process dictionary :  {no_tta,with_tta} -> {attribute 1: [task0, task1 etc...], attribute 2: [task0,
    # task1 etc...]}
    for n, (dataset, dataset_results) in enumerate(raw_test_results.items()):
        # ('no_tta', 'with_tta')
        for tta_mode in dataset_results[0].keys():
            if n == 0:
                all_results[tta_mode]={}
            for key in dataset_results[0][tta_mode].keys():
                if n == 0:
                    all_results[tta_mode][key]=[]
                all_results[tta_mode][key].extend([task_results[tta_mode][key] for task_results in dataset_results])
    # split and save results
    for (tta_mode, tta_mode_results) in all_results.items():
        # make dir
        tta_mode_test_results_dir = test_results_dir + f'/{tta_mode}'
        if not os.path.exists(tta_mode_test_results_dir):
            os.makedirs(tta_mode_test_results_dir)
        # save test results
        with open(f'{tta_mode_test_results_dir}/test_results.pickle', 'wb') as handle:
            pickle.dump({tta_mode:tta_mode_results}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # save task phi
        with open(f'{tta_mode_test_results_dir}/task_phis.pickle', 'wb') as handle:
            dict_to_write={tta_mode:raw_phi_results.pop(tta_mode+'_phi'),
                         'OOD':raw_phi_results.get('OOD'),
                           'prototypes': raw_phi_results.get(tta_mode + 'prototypes'),
                           'features': raw_phi_results.get(tta_mode + 'features'),
                           'labels':raw_phi_results.get(tta_mode+'labels'),
                        's_prototypes':raw_phi_results.get(tta_mode+'s_prototypes'),
                        'sq_pseudo_prototypes':raw_phi_results.get(tta_mode+'sq_pseudo_prototypes')}
            pickle.dump(dict_to_write, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log_str += f'Updated raw results and phis for [{tta_mode}]\n'
        # check and replace nan values for analysis
        for key, item in tta_mode_results.items():
            if type(item[0]) == float:
                nan_entry = np.isnan(item)
                if nan_entry.sum():
                    log_str += f'[{tta_mode}][{key}], contains {nan_entry.sum()} nan entries\n'
                    for idx in np.arange(len(item))[nan_entry]:
                        item[idx] = 0.0

    # plotting for ood_scores histograms, ood_scores vs acc etc.. and tta_trajectories for tta methods
    all_results_copy = copy.deepcopy(all_results)
    filter_mode = 'no_tta'
    base_results = all_results_copy.get(filter_mode)
    accuracy_table = OrderedDict({})

    for tta_mode in all_results_copy.keys():
        root_test_results_dir = test_results_dir + f'/{tta_mode}'
        tta_mode_log_str =""
        tta_mode_log_str += f'{f"{tta_mode}".upper() :=^50}\n'
        accuracy_table.update({tta_mode: []})

        if tta_mode != filter_mode:
            # extra_keys=[key for key in all_results_copy[tta_mode].keys() if key not in all_results_copy[filter_mode].keys()]
            # tta_records_only.update({'OOD':all_results[tta_mode]['OOD']})
            _plot_tta_trajectory_pdf(all_results_copy[tta_mode], root_test_results_dir)
            if base_results and not hps:
                tta_mode_log_str += _plot_tta_deltaacc_pdf(all_results_copy[tta_mode],base_results,root_test_results_dir)
            # tta_records_only = {key: all_results_copy[tta_mode].pop(key) for key in extra_keys}

        results_df = pd.DataFrame.from_dict(all_results_copy[tta_mode]) # convert to pandas.dataframe
        if not hps:  #todo: enable when not running hps
            tta_mode_log_str += _plot_auroc_pdf(results_df, ID_datasets)
            _plot_tasks_distribution_pdf(results_df, root_test_results_dir)
            _plot_ece_pdf(results_df, root_test_results_dir, ID_datasets)
            _plot_correlation_pdf(results_df, root_test_results_dir, ID_datasets)

        dfs = OrderedDict(tuple(results_df.groupby('OOD',sort=False)))
        for key, value in dfs.items():
            confidence = (1.96 * value['accuracy'].std()) / np.sqrt(len(value['accuracy'])) * 100
            mean_acc = value['accuracy'].mean() * 100
            accuracy_table[tta_mode].append(f'{mean_acc:3.2f} +/- {confidence:2.2f}')
        if not accuracy_table.get('dataset'):
            accuracy_table.update({'dataset':list(dfs.keys())})
        tta_mode_log_str+=f'{f"":=^50}\n\n'
        # log tta mode results
        tta_mode_logfile = open(f'{root_test_results_dir}/test_log.txt', "a+", buffering=1)
        print_and_log(tta_mode_logfile, tta_mode_log_str)
        tta_mode_logfile.close()

    # plot accuracy table for all test tta mode results in this round:
    accuracy_table.move_to_end('dataset', last=False)
    log_str += '\n'
    log_str += tabulate(accuracy_table, headers="keys", tablefmt="grid")
    log_str += '\n'
    return log_str


