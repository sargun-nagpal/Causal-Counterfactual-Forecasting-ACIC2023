# Copyright (c) 2023, Sargun Nagpal

import pickle
import numpy as np
from utils.evaluation_utils import load_trained_model


def process_counterfactual_seq_test_data(data_map, states, projection_horizon):
    sequence_lengths = data_map['sequence_lengths']
    current_treatments = data_map['current_treatments']
    previous_treatments = data_map['previous_treatments']
    current_covariates = data_map['current_covariates']

    num_units = current_covariates.shape[0]
    seq2seq_state_inits = np.zeros((num_units, states.shape[-1])) # (392, d)
    seq2seq_previous_treatments = np.zeros((num_units, projection_horizon, previous_treatments.shape[-1])) # (392 units, 5 steps, 6 treatments)
    seq2seq_current_treatments = np.zeros((num_units, projection_horizon, current_treatments.shape[-1])) # (392, 5, 6)
    seq2seq_current_covariates = np.zeros((num_units, projection_horizon, current_covariates.shape[-1])) # (392, 5, 7-outcome+static)
    seq2seq_active_entries = np.zeros((num_units, projection_horizon, 1))
    seq2seq_sequence_lengths = np.zeros(num_units)

    for i in range(num_units): # For each unit
        seq_length = int(sequence_lengths[i])
        seq2seq_state_inits[i] = states[i, seq_length - 1] # Latest encoder history state
        seq2seq_active_entries[i] = np.ones(shape=(projection_horizon, 1))
        seq2seq_previous_treatments[i] = previous_treatments[i, seq_length - 1:seq_length + projection_horizon - 1, :] # Treatment 90-94
        seq2seq_current_treatments[i] = current_treatments[i, seq_length:seq_length + projection_horizon, :] # Treatment 91-95
        seq2seq_sequence_lengths[i] = projection_horizon
        seq2seq_current_covariates[i] = np.repeat([current_covariates[i, seq_length - 1]], projection_horizon, axis=0) # Covariates (static) x5

    # Package outputs
    seq2seq_data_map = {
        'init_state': seq2seq_state_inits, # Encoder H(t): (392, d)
        'previous_treatments': seq2seq_previous_treatments, # (392, 5, 6) Treatments 90-94
        'current_treatments': seq2seq_current_treatments,  # (392, 5, 6) Treatments 91-95
        'current_covariates': seq2seq_current_covariates, # (392, 5, 7) Lates rec of covariates repeated 5 times (static)
        'sequence_lengths': seq2seq_sequence_lengths,
        'active_entries': seq2seq_active_entries,
        'outputs': np.ones(1) # Just to  initialize model in load_trained_model
        #'unscaled_outputs': seq2seq_outputs
    }
    return seq2seq_data_map


def duplicate_encoder_data(encoder_arr, copies):
    """ 
    Duplicating acc to ACIC data requirement. For each unit, we have 6 possible treatment plans: 0-0-0-0-0, ..., 5-5-5-5-5.
    Each of these treatment plans need the same encoder initialization. Hence duplicating encoder data. 
    """
    dup_arr = np.array([encoder_arr[0] for _ in range(copies)])
    for i in range(1, encoder_arr.shape[0]): # Repeat for each unit
        temp_arr = np.array([encoder_arr[i] for _ in range(copies)])
        dup_arr = np.append(dup_arr, temp_arr, axis=0)
    return dup_arr


def infer_CRN(data_map, projection_horizon, models_dir,
                     encoder_model_name, encoder_hyperparams_file,
                     decoder_model_name, decoder_hyperparams_file
                     ):
    # Test data for decoder
    test_processed_enc = data_map['test_data'] # For getting encoder output (all covariates)
    test_processed_dec = data_map['test_data_seq'] # For decoder (outcome + static covariates)

    # Encoder representation
    encoder_model = load_trained_model(test_processed_enc, encoder_hyperparams_file, encoder_model_name,
                                       models_dir)
    test_br_states = encoder_model.get_balancing_reps(test_processed_enc) # Encoder states (num_units, 94 steps, 12 covariates+treatments)
    test_br_outputs = encoder_model.get_predictions(test_processed_enc) # Encoder outputs (Y_t+1) -- Input to decoder (num_units, 94 steps, 1)
    
    # Duplicate encoder data
    test_br_states = duplicate_encoder_data(test_br_states, 6)
    test_br_outputs = duplicate_encoder_data(test_br_outputs, 6)
    print("-"*50)
    print(test_br_states.shape, test_br_outputs.shape)
    print("Produced Encoder output for Inference data")

    test_seq_processed = process_counterfactual_seq_test_data(test_processed_dec, test_br_states,
                                                              projection_horizon)    
    CRN_decoder = load_trained_model(test_seq_processed, decoder_hyperparams_file, decoder_model_name, models_dir,
                                      b_decoder_model=True)
    seq_predictions = CRN_decoder.get_autoregressive_sequence_predictions(test_processed_dec,
                                                                           test_br_states, test_br_outputs,
                                                                           projection_horizon)
    with open('infer_preds.p', 'wb') as f:
        pickle.dump(seq_predictions, f, protocol=pickle.HIGHEST_PROTOCOL)