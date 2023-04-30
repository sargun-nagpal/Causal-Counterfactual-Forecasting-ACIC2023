# Copyright (c) 2020, Ioana Bica

import os
import pickle5 as pickle
import argparse
import logging

from CRN_encoder_evaluate import test_CRN_encoder
from CRN_decoder_evaluate import test_CRN_decoder
from CRN_inference import infer_CRN


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default='results')
    parser.add_argument("--model_name", default="crn_test_2")
    parser.add_argument("--b_encoder_hyperparm_tuning", default=False)
    parser.add_argument("--b_decoder_hyperparm_tuning", default=False)
    return parser.parse_args()

def load_pickle(filename):
    with open(f'data/{filename}.p', 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':

    args = init_arg()

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    train_enc = load_pickle("train_enc")
    val_enc = load_pickle("val_enc")
    test_enc = load_pickle("test_enc")
    train_dec = load_pickle("train_dec")
    val_dec = load_pickle("val_dec")
    test_seq_dec = load_pickle("test_seq_dec")

    print("Loaded Data")
    pickle_map_enc = {}
    pickle_map_enc['training_data'] = train_enc
    pickle_map_enc['validation_data'] = val_enc
    pickle_map_enc['test_data'] = test_enc

    pickle_map_dec = {}
    pickle_map_dec['training_data'] = train_dec
    pickle_map_dec['validation_data'] = val_dec
    pickle_map_dec['test_data_seq'] = test_seq_dec

    encoder_model_name = 'encoder_' + args.model_name
    encoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, encoder_model_name)

    models_dir = '{}/crn_models'.format(args.results_dir)
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    print("Running Encoder")
    rmse_encoder = test_CRN_encoder(pickle_map=pickle_map_enc, models_dir=models_dir,
                                    encoder_model_name=encoder_model_name,
                                    encoder_hyperparams_file=encoder_hyperparams_file,
                                    b_encoder_hyperparm_tuning=args.b_encoder_hyperparm_tuning)


    decoder_model_name = 'decoder_' + args.model_name
    decoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, decoder_model_name)

    """
    The counterfactual test data for a sequence of treatments in the future was simulated for a 
    projection horizon of 5 timesteps. 
   
    """

    max_projection_horizon = 5
    projection_horizon = 5
    
    rmse_decoder = test_CRN_decoder(pickle_map_encoder=pickle_map_enc,
                                    pickle_map_decoder=pickle_map_dec,
                                    max_projection_horizon=max_projection_horizon,
                                    projection_horizon=projection_horizon,
                                    models_dir=models_dir,
                                    encoder_model_name=encoder_model_name,
                                    encoder_hyperparams_file=encoder_hyperparams_file,
                                    decoder_model_name=decoder_model_name,
                                    decoder_hyperparams_file=decoder_hyperparams_file,
                                    b_decoder_hyperparm_tuning=args.b_decoder_hyperparm_tuning)

    print("RMSE for one-step-ahead prediction.")
    print(rmse_encoder)

    print("Results for 5-step-ahead prediction.")
    print(rmse_decoder)

    inference_test = load_pickle("inf_enc")
    inference_test_seq = load_pickle("inf_seq_dec")
    print("Loaded Inference Data")

    pickle_map_infer = {}
    pickle_map_infer['test_data'] = inference_test
    pickle_map_infer['test_data_seq'] = inference_test_seq
    
    infer_CRN(data_map=pickle_map_infer,
            projection_horizon=projection_horizon,
            models_dir=models_dir,
            encoder_model_name=encoder_model_name,
            encoder_hyperparams_file=encoder_hyperparams_file,
            decoder_model_name=decoder_model_name,
            decoder_hyperparams_file=decoder_hyperparams_file)