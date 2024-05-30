# Project-transformers

## Description
In this work, we evaluated Deocder-only text generative models' ranking abilities, by measuring their AUROC across varying entities.
We define the term 'concept' and make the evaluation using that granularity of text segment.

## Install
0. Optional - It is recommended to create an environment. One can use Venv or Conda for this.
   
2. Clone this repository and install requirements:
```
git clone https://github.com/IdoAmit198/Project-transformers.git
pip install -r requirements.txt
```
2. clone the modified FActScore repository and install its requirements:
```
git clone -b working https://github.com/IdoAmit198/FActScore-Ido.git
cd FActScore-Ido && pip install .
pip install -r requirements.txt
```
The FActScore requirements consist of an un-updated version of OpenAI. Thus, we modified the files and so we require the newer OpenAI version:
```
pip install -U openai
```
3. Follow the `Download the data` section in the `FActScore-Ido` repository, to download the Wikidata database.

Warning: The database requires about ~22GB of space and might take time to download.

4. Keys: Our method was tested while utilizing GPT-4o model of OpenAI. To use OpenAI models, insert your API-Key in the file `my_open_ai_key.txt`. Alternatively, one can use open-source model by modifying our calls to OpenAI in the two scripts `data_pipeline.py` and `uncertainty_estimation.py`, and call FActScore with Llama instead of GPT-4. Additionaly, to reproduce our results, or even use Llama as an alternative to OpenAI models, you should login to HuggingFace hub. To do so, insert your HuggingFace API-key in the file `my_HF_key.txt`. More details are at the bottom.

## How to run
1. First, you would like to process the data. For that purpose, we've created the script `data_pipeline.py`, which does the following:
   1. Extract entities from the database.
   2. Given a subjetced LLM, generate passages about each entity.
   3. Segment the passages into concepts, in two steps. The first is manual to split into sentences while the second uses GPT-4o to segment a sentence into concepts, based on our definition.
   4. Use selective prediction ontop of FactScore for factuality labeling the generated concepts.

To run, first, modify the desired model and output path in the `data_pipeline.py` script in the main() function.

Then, run the following:
```
python data_pipeline.py
```

2. Once the data is ready, you would like to measure the model's ranking ability. For that purpose, we've created the `uncertainty_estimation.py` script. It does the following:
   1. For each generated concept, we measure the subjected LLM confidence in each token.
   2. We estimate different confidence scores using different confidence score functions.
   3. We measure the AUROC w.r.t each different confidence score and report it.

To run, first, modify the desired model and output path in the `uncertainty_estimation.py` script in the main() function.

Then, run the following:
```
python uncertainty_estimation.py
```
### Warnings:
1. In an attempt to reproduce our results, when running with `Mixtral-8x7B-Instruct-v0.1` model, one should have at least 120GB of GPU RAM. The code should work in a distributed manner, as long as you have enough space.
2. To access the `Mixtral-8x7B-Instruct-v0.1` and `Meta-Llama-3-8B-Instruct` models, one must have permission from HuggingFace. The access request can be made on the model pages. Once you have access, follow HuggingFace instructions to log in to your account in the working environment.
