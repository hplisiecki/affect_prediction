import torch
import numpy as np
from repo.dataset_and_model import Dataset

###############################################################################
"""
Utility functions for evaluation.
"""
###############################################################################

def evaluate(model, tokenizer, test_data, max_len, metric_names):
    """
    Evaluate the model on the test set.
    :param model: the model to evaluate
    :param tokenizer: the tokenizer to use
    :param test_data: the test data
    :param max_len: the maximum length of the sequences
    :param metric_names: the names of the metrics to use
    :return: the predictions and the labels
    """
    test = Dataset(tokenizer, test_data, max_len, metric_names)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    with torch.no_grad():
        for dataloader_pack in test_dataloader:

            input_mask_packs = sum(
                [[train_input['input_ids'].squeeze(1).to(device), train_input['attention_mask'].to(device)] for train_input in
                 dataloader_pack[:-1]], [])

            labels = dataloader_pack[-1]

            outputs = model(*input_mask_packs)

            try:
                for i in range(len(predictions_list)):
                    predictions_list[i].extend([o.cpu().detach().numpy() for o in outputs[i]])
                    labels_list[i].extend([l.numpy() for l in labels[i]])

            except:
                predictions_list = [[p for p in o.cpu().detach().numpy()] for o in outputs]
                labels_list = [[l for l in l.numpy()] for l in labels]

    return predictions_list, labels_list

def get_metrics(predictions, labels, metric_names):
    """
    Calculate correlations between the predictions and the labels.
    :param predictions: the predictions
    :param labels: the labels
    :param metric_names: the names of the metrics to use
    :return: the correlations
    """
    correlations = []
    for p,l,n in zip(predictions, labels, metric_names):
        p = [float(i) for i in p]
        l = [float(i) for i in l]
        corr = np.corrcoef(p, l)[0][1]
        print(f'Correlation for {n}: {corr}')
        correlations.append(corr)

    return correlations
