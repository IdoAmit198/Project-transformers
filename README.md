# Project-transformers

## Description
In this work, we evaluated Deocder-only text generative models' ranking abilities, by measuring their AUROC across varying entities.
We define the term 'concept' and make the evaluation using that granularity of text segment.

## Install

## How to run
1. First, you would like to process the data. For that purpose, we've created the script `data_pipeline.py`, which does the following:
   1. Extract entities from the database.
   2. Given a subjetced LLM, generate passages about each entity.
   3. Segment the passages into concepts, in two steps. The first is manual to split into sentences while the second uses GPT-4o to segment a sentence into concepts, based on our definition.

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
