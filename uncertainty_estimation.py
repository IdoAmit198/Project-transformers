
import json
import os
import pandas as pd
import numpy as np
import torch
from data_pipeline import get_titles

import accelerate

from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle

from huggingface_hub import login

###
# Methodology of our unceretainty estimation 
# 0. Collect entities from the DB.
# 1. For each entity, we will prompt a subjected model from HF to generate passages about it.
# 2. Each passage will be separated into atomic concepts, using GPT, or alternatively another instruct model.
# 3. Using the selective FactScore, we will label the concepts as factual or not.
#    Concepts with low certainty will not be tagged and will be discarded.
# 4. Next, we will prepend a prompt to each concept and assess the subjected model's certainty in the concept w.r.t the questions,
#    as described below.
###


###
# For each concept we all prepend a prompt pattern which query then likelihood of the generated concept for a given question
# For example:
# "Given the following question: {question}, how likely is the following concept: {concept}?"
# Next, we will use the tokens' probabilities to estimate the uncertainty of the concept, using various metrics (min, mean...)
###

def prompt_prepend(concept:str, title:str):
    question = f"Tell me about {title}"
    return f"Given the following question: {question}, tell me how likely is the following sentence: {concept} is."

def save_logprobs(model, tokenizer, concepts_path, title):
    df  = pd.read_csv(concepts_path)
    facts_logprob_dict = {}
    
    for idx,(_, row) in enumerate(df.iterrows()):
        input = prompt_prepend(row['concept'], title)
        concept_logprobs = to_tokens_and_logprobs(model, tokenizer, [input], [' '+row['concept']])
        facts_logprob_dict[row['concept']] = concept_logprobs[0] # Save a list of tuples (token, logprob).
    # Save facts_logprob_dict to pkl file
    # if directory does not exist, create it
    if not os.path.exists(f'logprobs/{model_name}'):
        os.makedirs(f'logprobs/{model_name}')
    with open(f'logprobs/{model_name}/{title}.pkl', 'wb') as f:
        pickle.dump(facts_logprob_dict, f)
    #load from 'facts_logprob_dict.pkl' into dict named d:
    # with open('facts_logprob_dict.pkl', 'rb') as f:
    #     d = pickle.load(f)
    #     print(d)
    #     print("finished")

def to_tokens_and_logprobs(model, tokenizer, input_texts, atoms_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids.to(model.device)
    atoms_ids = tokenizer(atoms_texts, padding=True, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    # Search for the beginning of atoms indices inside input_ids
    atoms_start = []
    atoms_end = []
    for input_row, atom_row in zip(input_ids, atoms_ids):
        found=0
        for idx in range(len(input_row) - len(atom_row) + 1):
            if (input_row[idx: idx + len(atom_row)] == atom_row.to(input_row.device)).all():
                atoms_start.append(idx)
                atoms_end.append(idx+len(atom_row))
                found=1
                break
        if found==0:
            for idx in range(len(input_row) - len(atom_row) + 1):
                if (input_row[idx+1: idx + len(atom_row)] == atom_row[1:].to(input_row.device)).all():
                    atoms_start.append(idx)
                    atoms_end.append(idx+len(atom_row))
                    break
    probs = probs[:, :-1, :]
    input_ids_ = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids_[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs, atom_start, atom_end in zip(input_ids_, gen_probs, atoms_start, atoms_end):
        text_sequence = []
        for token_idx, (token, p) in enumerate(zip(input_sentence, input_probs)):
            if token not in tokenizer.all_special_ids and token_idx>=atom_start-1 and token_idx<atom_end-1:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    assert len(batch)>0
    return batch

"""
Example: samples_certainties[0][0] is the confidence score of the first sample.
samples_certainties[0][1] is the correctness (True \ False) of the first sample.
"""
def AUROC(confidence, ground_truth, sort=True):
    samples_certainties = torch.stack([torch.tensor(confidence), torch.tensor(ground_truth)], dim=1)
    if sort:
        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        samples_certainties = samples_certainties[indices_sorting_by_confidence]
    total_samples = samples_certainties.shape[0]
    incorrect_after_me = np.zeros((total_samples))

    for i in range(total_samples - 1, -1, -1):
        if i == total_samples - 1:
            incorrect_after_me[i] = 0
        else:
            incorrect_after_me[i] = incorrect_after_me[i + 1] + (1 - int(samples_certainties[i + 1][1]))
            # Note: samples_certainties[i+1][1] is the correctness label for sample i+1

    n_d = 0  # amount of different pairs of ordering
    n_c = 0  # amount of pairs with same ordering
    incorrect_before_me = 0
    for i in range(total_samples):
        if i != 0:
            incorrect_before_me += (1 - int(samples_certainties[i - 1][1]))
        if samples_certainties[i][1]:
            # if i'm correct at this sample, i agree with all the incorrect that are to come
            n_c += incorrect_after_me[i]
            # i disagree with all the incorrect that preceed me
            n_d += incorrect_before_me
        else:
            # else i'm incorrect, so i disagree with all the correct that are to come
            n_d += (total_samples - i - 1) - incorrect_after_me[i]  # (total_samples - i - 1) = all samples after me
            # and agree with all the correct that preceed me
            n_c += i - incorrect_before_me

    smoothing = 0
    if (n_c + n_d) == 0:
        smoothing = 0.000001
    AUROC = (n_c) / (n_c + n_d + smoothing)
    return AUROC

def compute_uncertainty_metrics(concepts_logprobs_dict:dict, concepts_file_path:str):
    df = pd.read_csv(concepts_file_path)
    stop_words = ['a', 'an', 'the', 'is', 'are', 'was', 'were', 'has', 'have', 'had', 'been', 'being', 'be', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'as', 'by', 'about', 'between', 'from', 'into', 'through']
    for idx,(_, row) in enumerate(df.iterrows()):
        concept = row['concept']
        logprobs_packed = concepts_logprobs_dict[concept]
        logprobs = [logprob for token, logprob in logprobs_packed]
        logprobs_no_stopwords = [logprob for token, logprob in logprobs_packed if token.strip().lower() not in stop_words]
        for k, _logprobs in {'logprobs':logprobs, 'logprobs_no_stopwords':logprobs_no_stopwords}.items():
            _logprobs = np.array(_logprobs)
            probs = np.exp(_logprobs)
            s = '' if k == 'logprobs' else '_no_stopwords'
            df.at[idx, f'unc_mean_logprobs{s}'] = np.mean(_logprobs)
            df.at[idx, f'unc_mean_probs{s}'] = np.mean(probs)
            df.at[idx, f'unc_min_prob{s}'] = np.min(probs)
            df.at[idx, f'unc_min_logprob{s}'] = np.min(_logprobs)
            df.at[idx, f'unc_median_prob{s}'] = np.median(probs)
            df.at[idx, f'unc_median_logprobs{s}'] = np.median(_logprobs)
            df.at[idx, f'unc_PPL{s}'] = np.exp(-np.mean(_logprobs))
            df.at[idx, f'unc_Inv_PPL{s}'] = 1/(np.exp(-np.mean(_logprobs)))
            # df.at[idx, f'unc_PPL{s}'] = np.exp(-np.sum(probs*_logprobs))
            df.at[idx, f'unc_var_prob{s}'] = np.var(probs)
            df.at[idx, f'unc_var_logprob{s}'] = np.var(_logprobs)
            df.at[idx, f'unc_Inv_entropy{s}'] = 1/(-np.sum(probs*_logprobs))
            df.at[idx, f'unc_Inv_entropy_normalized_v2{s}'] = np.sqrt(len(_logprobs))/(-np.sum(probs*_logprobs))
            df.at[idx, f'unc_Inv_entropy_normalized{s}'] = 1/(-np.mean(probs*_logprobs))
            df.at[idx, f'unc_seq_len{s}'] = len(_logprobs)
            df.at[idx, f'unc_log_likelihood{s}'] = np.sum(_logprobs)
            df.at[idx, f'unc_sqrt_normalized_log_likelihood{s}'] = np.sum(_logprobs)/np.sqrt(len(_logprobs))

    df.to_csv(concepts_file_path, index=False)
    return df
    
def concate_and_compute(dir_path:str):
    # Concatenate all the csv files in the directory as a single dataframe and compute the AUROC from the unc columns
    df = pd.concat([pd.read_csv(f'{dir_path}/{f}') for f in os.listdir(dir_path) if f.endswith('.csv')])
    df.reset_index(inplace=True)
    AUROC_dict = {}
    for metric in [c for c in df.columns if 'unc' in c]:
        AUROC_dict[metric] = AUROC(df[metric], df['ground_truth'])
    # Save AUROC_dict to json file and break line for each element:
    # auroc_dir = 'AUROC_results'
    # if not os.path.exists(f'{auroc_dir}/{model_name}'):
    #     os.makedirs(f'{auroc_dir}/{model_name}')
    with open(f'Mixtral_AUROC.json', 'w') as f:
        json.dump(AUROC_dict, f, indent=4)
                

if __name__ == '__main__':

    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # concate_and_compute(f'selective_concepts/{model_name}')
    # exit()
    HF_key = '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'
    login(HF_key)
    titles = get_titles()
    # Compute logprobs for concepts in each title
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logprobs_dict_name = 'logprobs'
    for title in tqdm(titles, desc=f'Compute logprobs for titles'):
        # Check whether the pkl file logprobs for title exist already
        if os.path.exists(f'{logprobs_dict_name}/{model_name}/{title}.pkl'):
            print(f"The file {logprobs_dict_name}/{model_name}/{title}.pkl is already exist and processed. Moving on.")
            continue 
        if os.path.exists(f'selective_concepts/{model_name}/{title}.csv'):
            save_logprobs(model=model, tokenizer=tokenizer, concepts_path=f'selective_concepts/{model_name}/{title}.csv', title=title)

    #         if 'unc_mean_logprobs' in selective_df.columns:
    #         print(f"The file selective_concepts/{title}.csv is already exist and processed. Moving on.")
    #         continue
    #     save_logprobs(model_name, concepts_path, 'facts_logprob_dict.pkl')


    # logprobs_path = 'logprobs/meta-llama/Meta-Llama-3-8B-Instruct/Osama bin Laden.pkl'
    for title in tqdm(titles, desc=f'Compute logprobs for titles'):
        if os.path.exists(f'{logprobs_dict_name}/{model_name}/{title}.pkl'):
            with open(f'{logprobs_dict_name}/{model_name}/{title}.pkl', 'rb') as f:
                logprobs_dict = pickle.load(f)
                assert(type(logprobs_dict)==dict), f"The loaded object from path {logprobs_dict_name}/{model_name}/{title}.pkl is not a dictionary."
                concepts_file_path = f'selective_concepts/{model_name}/{title}.csv'
                df = pd.read_csv(concepts_file_path)
                # Check whether there is a column starts with 'unc'
                # if not any([c for c in df.columns if 'unc' in c]):
                    # print(f"The title {title} already has uncertainty metrics computed. Moving on. Notice to verify that auroc is computed as well.")
                    # continue
                compute_uncertainty_metrics(logprobs_dict, concepts_file_path)
                AUROC_dict = {}
                concept_df = pd.read_csv(concepts_file_path)
                for metric in [c for c in concept_df.columns if 'unc' in c]:
                    AUROC_dict[metric] = AUROC(concept_df[metric], concept_df['ground_truth'])
                # Save AUROC_dict to json file and break line for each element:
                auroc_dir = 'AUROC_results'
                if not os.path.exists(f'{auroc_dir}/{model_name}'):
                    os.makedirs(f'{auroc_dir}/{model_name}')
                with open(f'{auroc_dir}/{model_name}/{title}.json', 'w') as f:
                    json.dump(AUROC_dict, f, indent=4)
                