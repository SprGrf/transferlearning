# encoding=utf-8
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
from pathlib import Path
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import seaborn as sns
import datetime
import pickle

parser = argparse.ArgumentParser(description='argument setting of network')
# backbone model
parser.add_argument('--all_datasets', type=str, nargs='+', default=['C24', 'selfBACK', 'PAMAP2', 'GOTOV', 'DSA', 'MHEALTH', 'HHAR'], 
                    choices=['C24', 'selfBACK', 'PAMAP2', 'GOTOV', 'DSA', 'MHEALTH', 'HHAR'],
                    help='list of dataset names')

parser.add_argument('--metric', type=str, nargs='+', default=['F1'], 
                    choices=['F1', 'precision', 'recall'],
                    help='The metric to use for the main plot')

parser.add_argument('--results_dir', type=str, default=None, help='results directory')


# create directory for saving and plots
global plot_dir_name
plot_dir_name = 'plot/'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)


def boxplot(data):
    plt.boxplot(data, patch_artist=True)  
    plt.title("Box Plot Example")
    plt.xlabel("Dataset")
    plt.ylabel("Values")

    plt.show()

def multi_boxplot(model_names, results, title = None):
    """
    Plots a boxplot for each model with its results.
    
    Parameters:
    - model_names: A list of strings representing the names of the models.
    - results: A list of lists, where each sublist contains the results for a model.
    """
    if len(model_names) != len(results):
        raise ValueError("The number of model names must match the number of result lists.")
    
    # Creating the boxplot
    plt.figure(figsize=(10, 6))  
    plt.boxplot(results, patch_artist=True)  # Create box plots
    
    # Customizing the plot
    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel("Mean present class recall")
    plt.xticks(ticks=range(1, len(model_names) + 1), labels=model_names, rotation=45)
    
    # Optionally, add colors to distinguish boxes
    colors = ['lightblue', 'lightgreen', 'pink', 'lightyellow', 'orange']
    for patch, color in zip(plt.gca().artists, colors * len(results)):
        patch.set_facecolor(color)

    # Get the current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the plot with a filename including the current date and time
    plt.savefig(f'model_boxplot_{title}.png', dpi=300, bbox_inches='tight')


    # # Display the plot
    # plt.tight_layout()
    # plt.show()



def multi_violinplot(model_names, results):
    """
    Plots a violin plot for each model with its results, with annotations for mean, median, 
    and the number of datapoints.
    
    Parameters:
    - model_names: A list of strings representing the names of the models.
    - results: A list of lists, where each sublist contains the results for a model.
    """
    if len(model_names) != len(results):
        raise ValueError("The number of model names must match the number of result lists.")
    
    # Creating the violin plot
    plt.figure(figsize=(10, 6))  
    violin_parts = plt.violinplot(results, showmeans=True, showmedians=True)
    
    # Customizing the plot
    plt.title("Model Performance Violin Plot")
    plt.xlabel("Models")
    plt.ylabel("Results")
    plt.xticks(ticks=range(1, len(model_names) + 1), labels=model_names, rotation=45)
    
    # Optionally, add colors to distinguish violins
    colors = ['lightblue', 'lightgreen', 'pink', 'lightyellow', 'orange']
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])  # Cycle through colors
        pc.set_alpha(0.7)  # Set transparency
    
    # Customize means and medians
    violin_parts['cmeans'].set_color('blue')  # Color for the mean line
    violin_parts['cmeans'].set_linewidth(1.5)
    violin_parts['cmedians'].set_color('red')  # Color for the median line
    violin_parts['cmedians'].set_linewidth(1.5)
    
    # Annotate mean and median
    for i, data in enumerate(results, start=1):
        mean = sum(data) / len(data)
        median = sorted(data)[len(data) // 2]
        plt.text(i, mean, 'Mean', color='blue', fontsize=9, va='bottom', ha='left')
        plt.text(i, median, 'Median', color='red', fontsize=9, va='top', ha='left')
    
    # Annotate number of data points
    for i, data in enumerate(results, start=1):
        num_points = len(data)
        plt.text(i, max(data) * 1.05, f'n={num_points}', color='black', fontsize=10, ha='center')
    
    # Display the plot
    plt.tight_layout()
    plt.show()

def multimodel_boxplot(models, results_in2in, results_in2out, results_out2in, results_out2out, results_selftest, results_in2out_noisy, results_out_noisy2in, results_out_n2out_n, results_out_n2out, results_out2out_n, ncols=2):
    """
    Plots stacked vertical boxplots for each model, showing results for in2in and in2out.
    
    Parameters:
    - models: A list of strings representing the names of the models.
    - results_in2in: A list of lists, where each sublist contains the in2in results for a model.
    - results_in2out: A list of lists, where each sublist contains the in2out results for a model.
    - results_out2in: A list of lists, where each sublist contains the out2in results for a model.
    - results_out2out: A list of lists, where each sublist contains the out2out results for a model.
    - results_selftest: A list of lists, where each sublist contains the selftest results for a model.
    """
    if len(models) != len(results_in2in) or len(models) != len(results_in2out) or len(models) != len(results_out2in) or len(models) != len(results_out2out) or len(models) != len(results_selftest):
        raise ValueError("The number of models must match the number of result lists for both in2in and in2out.")
    
    # Creating the subplots
    num_models = len(models)
    nrows = (num_models + ncols - 1) // ncols  # Calculate the required number of rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6 * nrows))
    axes = axes.flatten()  # Flatten to make indexing easier 
    
    for i, (model, in2in, in2out, out2in, out2out, selftest, in2outn, outn2in, outn2outn, outn2out, out2outn) in enumerate(zip(models, results_in2in, results_in2out, results_out2in, results_out2out, results_selftest, results_in2out_noisy, results_out_noisy2in, results_out_n2out_n, results_out_n2out, results_out2out_n)):
        # Combine the two result sets for boxplots
        data = [selftest, in2in, in2out, out2in, out2out, in2outn, outn2in, outn2outn, outn2out, out2outn]
        
        # Plot on the corresponding subplot
        axes[i].boxplot(data, patch_artist=True, widths=0.6)
        axes[i].set_title(f"{model}")
        axes[i].set_ylabel("Results")
        axes[i].set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        axes[i].set_xticklabels(['SelfTest', 'I2I', 'I2O', 'O2I', 'O2O', 'I2ON', 'ON2I', 'ON2ON', 'ON2O', 'O2ON'], fontsize=5)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add color to the boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(axes[i].artists, colors):
            patch.set_facecolor(color)
    
    # Add common x-axis label and adjust layout
    plt.xlabel("Result Type")
    plt.tight_layout()
    plt.show()

def multimodel_violinplot(models, results_in2in, results_in2out, results_out2in, results_out2out, results_selftest, results_in2out_noisy, results_out_noisy2in, results_out_n2out_n, results_out_n2out, results_out2out_n, ncols=2):
    """
    Plots stacked vertical violin plots for each model, showing results for multiple test cases.
    Handles empty data arrays gracefully.
    
    Parameters:
    - models: A list of strings representing the names of the models.
    - results_in2in: A list of lists, where each sublist contains the in2in results for a model.
    - results_in2out: A list of lists, where each sublist contains the in2out results for a model.
    - results_out2in: A list of lists, where each sublist contains the out2in results for a model.
    - results_out2out: A list of lists, where each sublist contains the out2out results for a model.
    - results_selftest: A list of lists, where each sublist contains the selftest results for a model.
    - ncols: Number of columns in the subplot grid.
    """
    if len(models) != len(results_in2in) or len(models) != len(results_in2out) or len(models) != len(results_out2in) or len(models) != len(results_out2out) or len(models) != len(results_selftest):
        raise ValueError("The number of models must match the number of result lists for all test cases.")
    
    # Creating the subplots
    num_models = len(models)
    nrows = (num_models + ncols - 1) // ncols  # Calculate the required number of rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6 * nrows))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    for i, (model, in2in, in2out, out2in, out2out, selftest, in2outn, outn2in, outn2outn, outn2out, out2outn) in enumerate(zip(models, results_in2in, results_in2out, results_out2in, results_out2out, results_selftest, results_in2out_noisy, results_out_noisy2in, results_out_n2out_n, results_out_n2out, results_out2out_n)):
        # Combine the result sets for violin plots
        data = [selftest, in2in, in2out, out2in, out2out, in2outn, outn2in, outn2outn, outn2out, out2outn]
        labels = ['SelfTest', 'I2I', 'I2O', 'O2I', 'O2O', 'I2ON', 'ON2I', 'ON2ON', 'ON2O', 'O2ON']
        
        # Filter out empty data arrays for plotting
        non_empty_data = [d for d in data if len(d) > 0]
        non_empty_labels = [labels[j] for j, d in enumerate(data) if len(d) > 0]
        
        # Create the violin plot only if there is non-empty data
        if non_empty_data:
            violin_parts = axes[i].violinplot(non_empty_data, showmeans=True, showmedians=True)
            
            # Customize means and medians
            violin_parts['cmeans'].set_color('blue')  # Color for the mean line
            violin_parts['cmeans'].set_linewidth(1.5)
            violin_parts['cmedians'].set_color('red')  # Color for the median line
            violin_parts['cmedians'].set_linewidth(1.5)
            
            # Add color to the violin bodies
            colors = ['lightblue', 'lightgreen', 'pink', 'lightyellow', 'orange', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue']
            for j, pc in enumerate(violin_parts['bodies']):
                pc.set_facecolor(colors[j % len(colors)])  # Cycle through colors
                pc.set_alpha(0.7)  # Set transparency
            
            # Annotate mean, median, and number of points
            for j, category_data in enumerate(non_empty_data, start=1):
                mean = sum(category_data) / len(category_data)
                median = sorted(category_data)[len(category_data) // 2]
                axes[i].text(j, mean, 'Mean', color='blue', fontsize=9, va='bottom', ha='left')
                axes[i].text(j, median, 'Median', color='red', fontsize=9, va='top', ha='left')
                num_points = len(category_data)
                axes[i].text(j, max(category_data) * 1.05, f'n={num_points}', color='black', fontsize=10, ha='center')
        
        # Indicate missing data for categories with empty arrays
        for j, category_data in enumerate(data, start=1):
            if len(category_data) == 0:
                axes[i].text(j, 0.5, 'No data', color='gray', fontsize=9, ha='center')
        
        # Customize subplot
        axes[i].set_title(f"{model}")
        axes[i].set_ylabel("Results")
        axes[i].set_xticks(range(1, len(data) + 1))
        axes[i].set_xticklabels(labels, fontsize=5)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Remove empty subplots
    for j in range(len(models), len(axes)):
        fig.delaxes(axes[j])
    
    # Add common x-axis label and adjust layout
    plt.tight_layout()
    plt.show()

def remove_labels_from_confusion_matrix(cm, labels_to_omit):
    if len(labels_to_omit) > 0:
        print(cm)
        labels_to_omit_set = set(labels_to_omit)
        indices_to_keep = [i for i in range(cm.shape[0]) if i not in labels_to_omit_set]
        cm_filtered = cm[np.ix_(indices_to_keep, indices_to_keep)]
        print(cm_filtered)
        return cm_filtered
    else:
        return cm   

def collect_results(results_dir, models):
    for model in models:
        # print("Model is ", model)
        model_dir = Path(results_dir)/model
        
        ## Lists to store the results for three cases
        ## in2in: Training on DSA, testing on MHEALTH
        ## in2out: Training on DSA, testing on C24
        ## out2in: Training on C24, testing on DSA
        in2in_cms = []
        in2out_cms = []
        in2out_noisy_cms = []
        out2in_cms = []
        out_noisy2in_cms = []
        out_noisy2out_cms = []
        out2out_noisy_cms = []

        all_items = os.listdir(model_dir)
        datasets = [item for item in all_items if os.path.isdir(os.path.join(model_dir, item))] 
        print(datasets)
        for dataset in datasets:
            # print("Dataset is ", dataset)
            if dataset == 'GOTOV':
                omit_label = [4] # running
            elif dataset == 'selfBACK':
                omit_label = [5] # cycling
            elif dataset == 'HHAR':
                omit_label = [0, 4] # lying, running 
            else:
                omit_label = []

            dataset_dir = Path(model_dir)/dataset
            if dataset not in ['C24','C24n']:
                processedfileInList = os.listdir(dataset_dir)
                for file in processedfileInList:
                    if file.endswith(".cms"):
                        # print(file)
                        cms = np.load(Path(dataset_dir)/file, allow_pickle=True)
                        if "C24" not in file:
                            for cm in cms:
                                cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                                in2in_cms.append(cm_filtered)
                        elif "C24n"  in file:
                            for cm in cms:
                                cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                                in2out_noisy_cms.append(cm_filtered)                          
                        elif "C24"  in file:
                            for cm in cms:
                                cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                                in2out_cms.append(cm_filtered)      
            elif dataset == 'C24':
                processedfileInList = os.listdir(dataset_dir)
                for file in processedfileInList:
                    if file.endswith(".cms"):
                        # print(file)
                        cms = np.load(Path(dataset_dir)/file, allow_pickle=True)
                        if "C24n" in file:
                            # for cm in cms:
                            #     cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                            #     out2out_noisy_cms.append(cm_filtered)
                            print("Skipping this, will include CV results for cross C24 evaluation.")                           
                        else:
                            for cm in cms:
                                cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                                out2in_cms.append(cm_filtered)
            elif dataset == 'C24n':
                processedfileInList = os.listdir(dataset_dir)
                for file in processedfileInList:
                    if file.endswith(".cms"):
                        # print(file)
                        cms = np.load(Path(dataset_dir)/file, allow_pickle=True)
                        if "test_C24"  in file:
                            # for cm in cms:
                            #     cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                            #     out_noisy2out_cms.append(cm_filtered) 
                            print("Skipping this, will include CV results for cross C24 evaluation.")                           
                        else:
                            for cm in cms:
                                cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                                out_noisy2in_cms.append(cm_filtered)

    return in2in_cms, in2out_cms, out2in_cms, in2out_noisy_cms, out_noisy2in_cms, out_noisy2out_cms, out2out_noisy_cms

def collect_CV_results(results_dir, models):
    for model in models:
        print("Model is ", model)
        model_dir = Path(results_dir)/model

        all_items = os.listdir(model_dir)
        datasets = [item for item in all_items if os.path.isdir(os.path.join(model_dir, item))]         
        
        ## Self test: Training on DSA, testing on DSA with CV
        ## out2out: Training on C24, testing on C24 with CV
        self_test_cms = []
        out2out_cms = []
        outn2outn_cms = []

        for dataset in datasets:
            print("Dataset is ", dataset)
            if dataset == 'GOTOV':
                omit_label = [4] # running
            elif dataset == 'selfBACK':
                omit_label = [5] # cycling
            elif dataset == 'HHAR':
                omit_label = [0, 4] # lying, running 
            else:
                omit_label = []

            dataset_dir = Path(model_dir)/dataset
            if dataset not in ['C24','C24n']:
                processedfileInList = os.listdir(dataset_dir)

                cumulative_cms = [file for file in processedfileInList if file.endswith(".cms") and "round" not in file]
                if cumulative_cms:
                    # print("Files ending with '.cms' and not containing 'round':")
                    # print(cumulative_cms)
                    for file in cumulative_cms:
                            cms = np.load(Path(dataset_dir)/file, allow_pickle=True)
                            for cm in cms:
                                cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                                self_test_cms.append(cm_filtered) 
                else:
                    round_cms = [file for file in processedfileInList if file.endswith(".cms") and "round" in file]
                    if round_cms:
                        print("Files containing 'round' and ending with '.cms':")
                        print(round_cms)
                        for file in round_cms:
                            cms = np.load(Path(dataset_dir)/file, allow_pickle=True)
                            for cm in cms:
                                cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                                self_test_cms.append(cm_filtered) 
                    else:
                        print("No matching files found.")
   
            elif dataset == 'C24':                
                processedfileInList = os.listdir(dataset_dir)
                cumulative_cms = [file for file in processedfileInList if file.endswith(".cms") and "round" not in file]
                if cumulative_cms:
                    # print("Files ending with '.cms' and not containing 'round':")
                    # print(cumulative_cms)
                    for file in cumulative_cms:
                            cms = np.load(Path(dataset_dir)/file, allow_pickle=True)
                            for cm in cms:
                                cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                                out2out_cms.append(cm_filtered) 
                else:
                    round_cms = [file for file in processedfileInList if file.endswith(".cms") and "round" in file]
                    if round_cms:
                        print("Files containing 'round' and ending with '.cms':")
                        print(round_cms)
                        for file in round_cms:
                            cms = np.load(Path(dataset_dir)/file, allow_pickle=True)
                            for cm in cms:
                                cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                                out2out_cms.append(cm_filtered) 
                    else:
                        print("No matching files found.")
            elif dataset == 'C24n':                
                processedfileInList = os.listdir(dataset_dir)
                cumulative_cms = [file for file in processedfileInList if file.endswith(".cms") and "round" not in file]
                if cumulative_cms:
                    # print("Files ending with '.cms' and not containing 'round':")
                    # print(cumulative_cms)
                    for file in cumulative_cms:
                            cms = np.load(Path(dataset_dir)/file, allow_pickle=True)
                            for cm in cms:
                                cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                                outn2outn_cms.append(cm_filtered) 
                else:
                    round_cms = [file for file in processedfileInList if file.endswith(".cms") and "round" in file]
                    if round_cms:
                        print("Files containing 'round' and ending with '.cms':")
                        print(round_cms)
                        for file in round_cms:
                            cms = np.load(Path(dataset_dir)/file, allow_pickle=True)
                            for cm in cms:
                                cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)
                                outn2outn_cms.append(cm_filtered) 
                    else:
                        print("No matching files found.")
    
    return self_test_cms, out2out_cms, outn2outn_cms

def collect_cross_C24_results(results_dir, models):
    out_noisy2out_cms = []
    out2out_noisy_cms = []
    c24_variations = ['C24','C24n']

    for var in c24_variations:
        all_cms = []
        model_dir = Path(results_dir)/var

        for model in models:
            print("Model is ", model)
            dataset_dir = Path(model_dir)/model/var
        
            processedfileInList = os.listdir(dataset_dir)

            cumulative_cms = [file for file in processedfileInList if file.endswith(".cms") and "round" not in file]
            if cumulative_cms:
                # print("Files ending with '.cms' and not containing 'round':")
                # print(cumulative_cms)
                for file in cumulative_cms:
                        cms = np.load(Path(dataset_dir)/file, allow_pickle=True)
                        for cm in cms:
                            all_cms.append(cm) 
            else:
                round_cms = [file for file in processedfileInList if file.endswith(".cms") and "round" in file]
                if round_cms:
                    print("Files containing 'round' and ending with '.cms':")
                    print(round_cms)
                    for file in round_cms:
                        cms = np.load(Path(dataset_dir)/file, allow_pickle=True)
                        for cm in cms:
                            all_cms.append(cm) 
                else:
                    print("No matching files found.")

            if var == 'C24':
                out2out_noisy_cms = all_cms
            else:
                out_noisy2out_cms = all_cms

    return out_noisy2out_cms, out2out_noisy_cms

def calculate_multiclass_metrics(cm):
    # Initialize variables to store metrics for each class
    num_classes = cm.shape[0]
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_score_per_class = np.zeros(num_classes)

    # Calculate metrics for each class
    for i in range(num_classes):
        TP = cm[i, i]  # True Positives for class i
        FP = cm[:, i].sum() - TP  # False Positives for class i
        FN = cm[i, :].sum() - TP  # False Negatives for class i
        TN = cm.sum() - (TP + FP + FN)  # True Negatives for class i
       
        # Precision, Recall, F1 Score for class i
        precision_per_class[i] = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall_per_class[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score_per_class[i] = (2 * precision_per_class[i] * recall_per_class[i]) / (precision_per_class[i] + recall_per_class[i]) if (precision_per_class[i] + recall_per_class[i]) != 0 else 0


    # Calculate support for each class
    support = cm.sum(axis=1)  # Number of true instances for each class

    # Filter out classes with no true positives, which means they didn't appear in the test set
    precision_non_zero = precision_per_class[precision_per_class != 0]
    recall_non_zero = recall_per_class[recall_per_class != 0]
    f1_score_non_zero = f1_score_per_class[f1_score_per_class != 0]

    # Calculate macro-averaged metrics across classes that exist in the test set
    accuracy = np.trace(cm) / cm.sum()  # Overall accuracy
    precision_macro = precision_non_zero.mean() if precision_non_zero.size > 0 else 0
    recall_macro = recall_non_zero.mean() if recall_non_zero.size > 0 else 0
    f1_score_macro = f1_score_non_zero.mean() if f1_score_non_zero.size > 0 else 0

    # Calculate weighted F1 score
    weighted_f1 = sum(f1 * s for f1, s in zip(f1_score_per_class, support)) / support.sum()

    # Calculate mean recall for classes present in the data
    classes_present = np.where(support > 0)[0]  # Indices of classes with non-zero support
    mean_recall_present = recall_per_class[classes_present].mean() if len(classes_present) > 0 else 0

    # Calculate mean F1 including the "negative" class
    f1_scores_with_negative = list(f1_score_per_class[classes_present])  # Start with F1 scores for present classes
    for i, row_sum in enumerate(support):
        if row_sum == 0 and cm[:, i].sum() > 0:  # Negative class condition
            # TP = cm[i, i]
            # FP = cm[:, i].sum() - TP
            # precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            # f1_negative = 0  # Recall is undefined for these classes, so F1 is 0
            # f1_scores_with_negative.append(f1_negative)    #         

            # Don't really care to recalculate everything. If there is one class that fulfils this, append zero and exit
            f1_scores_with_negative.append(0)
            break

    mean_f1_with_negative = np.mean(f1_scores_with_negative) if f1_scores_with_negative else 0



    return {
        'accuracy': accuracy,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_score_per_class': f1_score_per_class,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_score_macro': f1_score_macro,
        'weighted_f1': weighted_f1,
        'mean_recall_present': mean_recall_present,
        'mean_f1_with_negative': mean_f1_with_negative
    }


def plot_comparison_boxplots(datasets, model_name, results_dir, num_classes=7):
    """
    Generate a KxK grid of boxplots comparing metrics across confusion matrices for each dataset combination.

    Parameters:
        datasets (list): List of dataset names (e.g., ['A', 'B', 'C', 'D']).
        model_name (str): Name of the model (used in file paths).
        results_dir (str): Path to the directory containing result files.
        metric_function (callable): Function to compute metrics from a confusion matrix.
        num_classes (int): Number of classes for the boxplots (default is 7).
    """
    K = len(datasets)  # Number of datasets
    fig, axes = plt.subplots(K, K, figsize=(4 * K, 4 * K), constrained_layout=True)
    
    for i, train_ds in enumerate(datasets):
        print(train_ds)
        for j, test_ds in enumerate(datasets):
            print(test_ds)
            ax = axes[i, j]
            
            # Leave diagonal empty
            if i == j:
                ax.axis("off")
                continue
            
            # Path to the confusion matrix result file
            file_path = os.path.join(results_dir, model_name, train_ds, f"{model_name}_{train_ds}_test_{test_ds}.cms")
            print(file_path)
            if os.path.exists(file_path):
                # Load list of confusion matrices
                confusion_matrices = np.load(file_path, allow_pickle=True)
                
                # Compute metrics for each confusion matrix
                metrics_by_class = [[] for _ in range(num_classes)]  # To collect metrics per class
                # print(metrics_by_class)
                for cm in confusion_matrices:
                    metrics = calculate_multiclass_metrics(cm)  # Compute metrics (e.g., F1 scores) per class
                    # print(metrics['f1_score_per_class'])
                    # print(metrics)
                    for cls in range(num_classes):
                        metrics_by_class[cls].append(metrics['f1_score_per_class'][cls])
                    # print(metrics_by_class)
                
                # Prepare data for boxplots
                ax.boxplot(metrics_by_class, positions=range(num_classes), patch_artist=True,
                           boxprops=dict(facecolor='lightblue', color='blue'))
                ax.set_title(f"Train: {train_ds}, Test: {test_ds}", fontsize=10)
                ax.set_xlabel("Class", fontsize=8)
                ax.set_ylabel("Metric Value", fontsize=8)
                ax.set_xticks(range(num_classes))
            else:
                # Handle missing files
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=10, color='red')
                ax.axis("off")
    
    # Adjust layout and display
    plt.suptitle(f"Model: {model_name} - Class Metric Boxplots", fontsize=16)
    plt.show()


def plot_comparison_matrix(datasets, model_name, results_dir, num_classes=7):
    """
    Generate a KxK grid representing the average F1 scores for each train-test dataset pair.
    The grid is color-coded, and missing data is indicated.

    Parameters:
        datasets (list): List of dataset names (e.g., ['A', 'B', 'C', 'D']).
        model_name (str): Name of the model (used in file paths).
        results_dir (str): Path to the directory containing result files.
        num_classes (int): Number of classes for the metrics (default is 7).
    """
    K = len(datasets)  # Number of datasets
    avg_recall_matrix = np.full((K, K), np.nan)  # Initialize matrix to store average F1 scores

    for i, train_ds in enumerate(datasets):
        for j, test_ds in enumerate(datasets):
            # Path to the confusion matrix result file
            file_path = os.path.join(results_dir, model_name, train_ds, f"{model_name}_{train_ds}_test_{test_ds}.cms")
            if os.path.exists(file_path):
                # Load list of confusion matrices
                confusion_matrices = np.load(file_path, allow_pickle=True)
                
                # Compute average F1 score for all confusion matrices
                mean_recall_scores = []
                for conm in confusion_matrices:
                    metrics = calculate_multiclass_metrics(conm)  # Compute metrics (e.g., F1 scores) per class
                    mean_recall_scores.append(metrics['mean_recall_present'])
                
                # Store the mean of all F1 scores
                avg_recall_matrix[i, j] = np.mean(mean_recall_scores)
    
    # Create the matrix plot
    fig, ax = plt.subplots(figsize=(6, 6))
    norm = Normalize(vmin=0, vmax=1)  # Normalize color scale to 0-1
    cmap = cm.Blues  # Use a color map (e.g., Blues)

    # Plot the matrix as a heatmap
    cax = ax.matshow(avg_recall_matrix, cmap=cmap, norm=norm)

    # Add values and labels to cells
    for i in range(K):
        for j in range(K):
            if np.isnan(avg_recall_matrix[i, j]):
                ax.text(j, i, "No Data", ha='center', va='center', color='red', fontsize=10)
            else:
                ax.text(j, i, f"{avg_recall_matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=10)
    
    # Configure axis labels
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(datasets, rotation=45, ha='left')
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Test Dataset", fontsize=12)
    ax.set_ylabel("Train Dataset", fontsize=12)
    ax.set_title(f"Model: {model_name} - Average present class recall", fontsize=14)

    # Add a color bar
    fig.colorbar(cax, ax=ax, orientation='vertical', label="Average Recall Score")
    plt.tight_layout()
    plt.show()



def plot_comparison_matrix_with_violins(datasets, model_name, results_dir, num_classes=7):
    """
    Generate a KxK grid representing the distributions of F1 scores for each train-test dataset pair.
    Each cell contains a violin plot showing the F1 scores per class.

    Parameters:
        datasets (list): List of dataset names (e.g., ['A', 'B', 'C', 'D']).
        model_name (str): Name of the model (used in file paths).
        results_dir (str): Path to the directory containing result files.
        num_classes (int): Number of classes for the metrics (default is 7).
    """
    K = len(datasets)  # Number of datasets
    f1_data_matrix = [[None for _ in range(K)] for _ in range(K)]  # Initialize storage for F1 score lists

    for i, train_ds in enumerate(datasets):
        for j, test_ds in enumerate(datasets):
            # Path to the confusion matrix result file
            file_path = os.path.join(results_dir, model_name, train_ds, f"{model_name}_{train_ds}_test_{test_ds}.cms")
            if os.path.exists(file_path):
                # Load list of confusion matrices
                confusion_matrices = np.load(file_path, allow_pickle=True)
                
                # Collect all F1 scores per class
                f1_scores_per_class = []
                for conm in confusion_matrices:
                    metrics = calculate_multiclass_metrics(conm)  # Compute metrics (e.g., F1 scores) per class
                    f1_scores_per_class.extend(metrics['f1_score_per_class'])
                
                # Store F1 scores for this cell
                f1_data_matrix[i][j] = f1_scores_per_class

    # Create the figure and axes for the matrix plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set up grid layout
    for i in range(K):
        for j in range(K):
            f1_scores = f1_data_matrix[i][j]
            cell_x = j  # Cell x-coordinate
            cell_y = i  # Cell y-coordinate (adjusted for top-to-bottom labeling)

            if f1_scores is None:
                # Plot a "No Data" text
                ax.text(cell_x + 0.5, K - cell_y - 0.5, "No Data", ha='center', va='center', color='red', fontsize=10)
            else:
                # Create an inset for the violin plot
                inset_ax = ax.inset_axes([cell_x / K + 0.05, (K - cell_y - 1) / K + 0.05, 1 / K - 0.1, 1 / K - 0.1])
                sns.violinplot(
                    y=f1_scores,
                    ax=inset_ax,
                    palette="Blues",
                    cut=0,
                    bw=0.2
                )
                # Remove inset axes' labels and ticks
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])
                inset_ax.set_xlabel("")
                inset_ax.set_ylabel("")

    # Configure axes
    ax.set_xlim(0, K)
    ax.set_ylim(0, K)
    ax.set_xticks([i + 0.5 for i in range(K)])
    ax.set_yticks([i + 0.5 for i in range(K)])
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(datasets[::-1], fontsize=10)  # Reverse the order of y-axis labels
    ax.set_xlabel("Test Dataset", fontsize=12)
    ax.set_ylabel("Train Dataset", fontsize=12)
    ax.set_title(f"Model: {model_name} - F1 Score Distributions", fontsize=14)

    # Add gridlines to separate cells
    for i in range(K + 1):
        ax.axhline(i, color='black', linewidth=0.5)
        ax.axvline(i, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def process_metrics(confusion_matrices):
    """
    Process a list of confusion matrices, calculate metrics, 
    and append the weighted F1 scores to the provided metrics list.
    
    Args:
        confusion_matrices (list): List of confusion matrices.
        metrics_list (list): List to store the weighted F1 scores.
        description (str): Description for debugging or logging.
    """
    metrics_list = []
    for conm in confusion_matrices:
        
        # print('///////////////////////////////////////////////////////////')
        # print(description, conm)
        metrics = calculate_multiclass_metrics(conm)
        # print(metrics['precision_per_class'])
        # print(metrics['recall_per_class'])
        # print(metrics['f1_score_per_class'])
        
        
        # metrics_list.append(metrics['weighted_f1'])

        # metrics_list.append(np.mean(metrics['f1_score_per_class']))
        metrics_list.append((metrics['mean_recall_present']))
        if metrics['mean_recall_present'] < 0.1:
            print("lala")
            print(conm)
        # metrics_list.append((metrics['mean_f1_with_negative']))
    return metrics_list
    
def search_and_load_files(parent_directory, target_filenames):
    cms = []
    
    for root, _, files in os.walk(parent_directory):
        for file in files:
            if file in target_filenames:
                file_path = os.path.join(root, file)
                print("loading ", file_path)
                with open(file_path, 'rb') as f:
                    conms = pickle.load(f)
                if 'GOTOV' in file_path:
                    omit_label = [4] # running
                elif 'selfBACK' in file_path:
                    omit_label = [5] # cycling
                elif 'HHAR' in file_path:
                    omit_label = [0, 4] # lying, running 
                else:
                    omit_label = []
                print(len(conms))
                for cm in conms:


                    cm_filtered = remove_labels_from_confusion_matrix(cm, omit_label)

                    cms.append(cm_filtered)       

    return cms

if __name__ == '__main__':
    args = parser.parse_args()
    in2in_models = []
    in2out_models = []
    out2in_models = []
    out2out_models = []
    self_test_models = []

    in2in_cms = search_and_load_files(args.results_dir, "in2in.cms")
    in2in_metrics = process_metrics(in2in_cms)

    in2out_cms = search_and_load_files(args.results_dir, "in2out.cms")
    in2out_metrics = process_metrics(in2out_cms)

    self_test_cms = search_and_load_files(args.results_dir, "self.cms")
    self_test_metrics = process_metrics(self_test_cms)

    out2in_cms = search_and_load_files(args.results_dir, "out2in.cms")
    out2in_metrics = process_metrics(out2in_cms)

    out2out_cms = search_and_load_files(args.results_dir, "out2out.cms")
    out2out_metrics = process_metrics(out2out_cms)


    # Organize data
    data = [self_test_metrics, in2in_metrics, in2out_metrics, out2in_metrics, out2out_metrics]
    labels = ["SelfTest", "In2In", "In2Out", "Out2In", "Out2Out"]
    
    pickle_dict = {}
    pickle_dict["in2in"] = in2in_metrics
    pickle_dict["self"] = self_test_metrics
    pickle_dict["in2out"] = in2out_metrics
    pickle_dict["out2in"] = out2in_metrics
    pickle_dict["out2out"] = out2out_metrics

    with open("FIXED" + "_metrics.pkl", "wb") as f:
        pickle.dump(pickle_dict, f)
    
    # Create boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels)
    plt.xlabel("Metric Type")
    plt.ylabel("Values")
    plt.title("Multi-boxplot of Metrics")
    plt.grid(True)

    # Show plot
    plt.show()

    # multimodel_boxplot(args.all_models, in2in_models, in2out_models, out2in_models, out2out_models, self_test_models, ncols=3)
    # multimodel_violinplot(args.all_models, in2in_models, in2out_models, out2in_models, out2out_models, self_test_models, ncols=3)

    #     # print("------------------------------------------------")
    #     # print(in2in_cms[0])
    #     # print(type(in2in_cms[0]))
    #     # print(len(in2in_cms[0]))    
    #     # print('**********************************************************')
    #     # print(type(in2in_cms[0][0]))    
    #     # print(in2in_cms[0][0])    

    # # plot_comparison_boxplots(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK'], 'FCN', 'results/all')
    # plot_comparison_matrix(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24', 'C24n'], 'FCN', args.results_dir)
    # plot_comparison_matrix(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24', 'C24n'], 'CNN_AE', args.results_dir)
    # plot_comparison_matrix(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24', 'C24n'], 'DCL', args.results_dir)
    # plot_comparison_matrix(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24', 'C24n'], 'LSTM', args.results_dir)
    # plot_comparison_matrix(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24', 'C24n'], 'Transformer', args.results_dir)
    # plot_comparison_matrix(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24', 'C24n'], 'AE', args.results_dir)
    # plot_comparison_matrix(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24', 'C24n'], 'RF', args.results_dir)
    # plot_comparison_matrix(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24', 'C24n'], 'KNN', args.results_dir)
    # plot_comparison_matrix(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24', 'C24n'], 'XGB', args.results_dir)
    # # plot_comparison_matrix_with_violins(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24'], 'KNN', args.results_dir)
    # # plot_comparison_matrix_with_violins(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24'], 'LSTM', args.results_dir)
    # # plot_comparison_matrix_with_violins(['MHEALTH', 'DSA', 'GOTOV', 'selfBACK', 'HHAR', 'PAMAP2', 'C24'], 'FCN', args.results_dir)
