import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from FActScore.factscore.factscorer import FactScorer

import os
import re

from FActScore.factscore.openai_lm import get_completion
from openai import OpenAI

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pickle

import time

def create_prompt(title:str, pipeline):
    if pipeline.model.config._name_or_path.startswith('mistralai/M'):
        messages = [
        {"role": "user", "content": f"Tell me about {title}"},
    ]
    else:
        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Tell me about {title}"},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True)
    
    return prompt

def generate_passages(model_name, titles, output_save_dir_path):
    # Load pipeline for model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token_id = "[PAD]"
    tokenizer.padding_side = "left"
    batch_size=20
    if model_name.startswith('mistralai/M') or model_name.startswith('mistralai/Mixtral'):
        generator = pipeline(model=model_name, tokenizer=tokenizer, task="text-generation", return_full_text=False,
                max_new_tokens=512, device_map="auto", trust_remote_code=True,
                batch_size=batch_size)
    else:
        generator = pipeline(model=model_name, tokenizer=tokenizer, task="text-generation", return_full_text=False,
                max_new_tokens=512, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True,
                batch_size=batch_size)
    generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
    terminators = [
        generator.tokenizer.eos_token_id,
        generator.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    not_exists_titles = []
    for title in titles:
        # Verify whether the file exists for each title
        file_path = f'{output_save_dir_path}/{title}.csv'
        if os.path.exists(file_path):
            continue
        else:
            not_exists_titles.append(title)
    # titles_prompts = [f"Tell me about {title}" for title in not_exists_titles]
    titles_generations = []
    titles_prompts = [create_prompt(title, generator) for title in not_exists_titles]
    # for title in not_exists_titles:
        # titles_generations.append(generator(prompt, eos_token_id=terminators))
    titles_generations = generator(titles_prompts,eos_token_id=terminators)
    return {title: generation[0]['generated_text'] for title, generation in zip(not_exists_titles, titles_generations)}

def split_to_sentences(generation:str):
    generation = generation.strip()
    sentences = [x.strip()+'.' for x in re.split("[//.|//!|//?]", generation) if x!=""]
    # print(sentences)
    # merge sentences that are too short with the previous sentence (if exists):
    for i in range(1, len(sentences)):
        if len(sentences[i])<10 and sentences[i] != '':
            sentences[i-1] += sentences[i]
            if i < len(sentences) - 1:
                # print(i)
                # print(sentences[i-1])
                # print(sentences[i])
                # print(sentences[i+1])
                sentences[i-1] += sentences[i+1]
            # remove the merged sentences
            sentences[i] = ""
            if i+1<len(sentences)-1:
                sentences[i+1] = ""
    sentences = [x for x in sentences if x!=""]
    return sentences

def GPT_segmentation(title, sentence:str) -> list[str]:
    # Call GPT to segment the sentence into concepts
    split_prompt = f"Break down the following sentence about {title} into independent facts: {sentence}"
    message = [{"role": "system", "content": "You have a vast general knowledge and an expert at segmentation of sentences to individual concepts.\
 A concept is a short standalone sentence, consisting of a single piece of information.\
 Write each concept in a new line. Avoid using propositions instead of nouns."},
               {"role": "user", "content": split_prompt}]
    response = get_completion(
                        message,
                        model="gpt-4o",
                        top_logprobs=None)
    response = response.choices[0].message.content
    concepts = []
    for concept in response.split('\n'):
        if concept != '':
            concepts += split_to_sentences(concept)
    return concepts


def segment_concepts(titles_generation_dict):
    titles_concepts_dict = {}
    for title, generation in titles_generation_dict.items():
        sentences = split_to_sentences(generation)
        title_concepts_list = []
        for sentence in sentences:
            # Call GPT to segment the sentence into concepts
            title_concepts_list += GPT_segmentation(title, sentence)
        # titles_concepts_dict[title] = concepts
        titles_concepts_dict[title] = title_concepts_list
    return titles_concepts_dict

def generate_concepts(model_name, titles, output_save_dir_path):
    """
    Given a model name, a list of titles and a path to a csv file:
    1. generate bios for each title.
    2. Segment each bio into atomic facts using GPT.
    3. Save the facts in the the csv file.
    4. return the facts
    """
    not_exists_titles = []
    for title in titles:
        # Verify whether the file exists for each title
        file_path = f'{output_save_dir_path}/{title}.csv'
        if os.path.exists(file_path):
            continue
        else:
            not_exists_titles.append(title)
    if len(not_exists_titles)>0:
        titles_generation_dict = generate_passages(model_name, not_exists_titles, output_save_dir_path)
        titles_concepts_dict = segment_concepts(titles_generation_dict)
        # Save the concepts to the csv file
        for title, concepts in titles_concepts_dict.items():
            concepts_df = pd.DataFrame(concepts, columns=['concept'])
            concepts_df['title'] = title
            concepts_df.to_csv(f'{output_save_dir_path}/{title}.csv', index=False)
    else:
        print(f"All titles have been processed already and are saved under {output_save_dir_path}.")

def save_confidence_scores(load_path:str, save_selective_path, title:str):
    df = pd.read_csv(load_path)
    # Convert the 'concept' column to a list of concepts
    concepts = df['concept'].tolist()
    # for idx, row in df.iterrows():
    #     concept = row['concept']
    # for concept,topic,gen in zip(concepts,titles,generations):
    OpenAi_key_path = "/home/ido.amit/Project-transformers/my_open_ai_key.txt"
    fs = FactScorer(openai_key=OpenAi_key_path, model_name="retrieval+ChatGPT")
    decisions = fs._get_score(title, " ", atomic_facts=concepts, theta=0)
    for (idx, row), decision in zip(df.iterrows(), decisions):
        assert row['concept'].strip() == decision['atom'].strip(), "The concepts are not aligned with the decisions."
        df.at[idx, 'true_score'] = decision['true_score']
        df.at[idx, 'false_score'] = decision['false_score']
        df.at[idx, 'ground_truth'] = decision['is_supported']
    # Add a column named 'confidence' which is the max between true and false scores
    df['confidence'] = df[['true_score', 'false_score']].max(axis=1)
    threshold = 0.95
    # Filter only rows with confidence higher than threshold
    df = df[df['confidence'] > threshold]
    df.to_csv(save_selective_path, index=False)

def get_titles():
    """
    return 20 titles in the DB as a list, from various genres, events, places and such.
    """
    return ["Osama bin Laden", "Barack Obama", "Donald Trump", "Freddie Mercury", "Elon Musk", "Albert Einstein", "Isaac Newton", 'The Beatles', 'World War II',  'American Civil War', "Leonardo da Vinci", "Pablo Picasso", "Vincent van Gogh", 'Industrial Revolution', "Marilyn Monroe", "Cleopatra", "Elizabeth II", "Napoleon", "Julius Caesar", "Alexander the Great"]

## Modify the code to create a unique csv file for each title, instead of the current code.
if __name__ == '__main__':
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    titles = get_titles()
    # facts_path = 'selective_facts.csv'
    # facts_path = 'selective_osama_facts_gpt_copy.csv'
    concepts_save_dir = f'concepts/{model_name}'
    if not os.path.exists(concepts_save_dir):
        os.makedirs(concepts_save_dir)
    generate_concepts(model_name, titles, concepts_save_dir)

    # Obtain ground truths:
    for title in tqdm(titles, desc=f'processing confidence for titles'):
        if os.path.exists(f'selective_concepts/{title}.csv'):
            print(f"The file selective_concepts/{title}.csv is already exist and processed. Moving on.")
            continue
        load_path = f'{concepts_save_dir}/{title}.csv'
        save_path = f'selective_concepts/{title}.csv'
        save_confidence_scores(load_path=load_path,save_selective_path=save_path , title=title)
        
        # print(df)

    print("Finished processing data")
    # Compute logprobs for concepts in each title
    # for title in tqdm(titles, desc=f'Compute logprobs for titles'):
    #     if os.path.exists(f'selective_concepts/{title}.csv'):
    #         selective_df = pd.read_csv(f'selective_concepts/{title}.csv')
    #         if 'unc_mean_logprobs' in selective_df.columns:
    #         print(f"The file selective_concepts/{title}.csv is already exist and processed. Moving on.")
    #         continue
    #     save_logprobs(model_name, concepts_path, 'facts_logprob_dict.pkl')
