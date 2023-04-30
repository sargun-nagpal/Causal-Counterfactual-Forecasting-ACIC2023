# Copyright (c) 2020, Ioana Bica

import numpy as np
import pandas as pd
from CRN_model import CRN_Model
import pickle


def write_results_to_file(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=2)

def append_results_to_file(filename, data):
    with open(filename, 'a+b') as handle:
        pickle.dump(data, handle, protocol=2)


def load_trained_model(dataset_test, hyperparams_file, model_name, model_folder, b_decoder_model=False):
    _, length, num_covariates = dataset_test['current_covariates'].shape
    num_treatments = dataset_test['current_treatments'].shape[-1]
    num_outputs = dataset_test['outputs'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 100}

    print("Loading best hyperparameters for model")
    with open(hyperparams_file, 'rb') as handle:
        best_hyperparams = pickle.load(handle)

    model = CRN_Model(params, best_hyperparams)
    if (b_decoder_model):
        model = CRN_Model(params, best_hyperparams, b_train_decoder=True)

    model.load_model(model_name=model_name, model_folder=model_folder)
    return model


def get_mse_at_follow_up_time(mean, output, active_entires):
        mses = np.sum(np.sum((mean - output) ** 2 * active_entires, axis=-1), axis=0) \
               / active_entires.sum(axis=0).sum(axis=-1)

        return pd.Series(mses, index=[idx for idx in range(len(mses))])


def train_BR_optimal_model(dataset_train, dataset_val, hyperparams_file, model_name, model_folder,
                           b_decoder_model=False):
    _, length, num_covariates = dataset_train['current_covariates'].shape
    num_treatments = dataset_train['current_treatments'].shape[-1]
    num_outputs = dataset_train['outputs'].shape[-1]

    params = {'num_treatments': num_treatments,
              'num_covariates': num_covariates,
              'num_outputs': num_outputs,
              'max_sequence_length': length,
              'num_epochs': 100}

    print("Loading best hyperparameters for model")
    with open(hyperparams_file, 'rb') as handle:
        best_hyperparams = pickle.load(handle)

    print("Best Hyperparameters")
    print(best_hyperparams)

    if (b_decoder_model):
        print(best_hyperparams)
        model = CRN_Model(params, best_hyperparams, b_train_decoder=True)
    else:
        model = CRN_Model(params, best_hyperparams)
    model.train(dataset_train, dataset_val, model_name=model_name, model_folder=model_folder)