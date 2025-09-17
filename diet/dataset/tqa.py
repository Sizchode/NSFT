import json
import os
import random

import pandas as pd
from datasets import Dataset


class TQA:
    def __init__(self, iti_split_dir, fold_num=0, data_gen_seed=42, prompt_template=None):
        self.iti_split_dir = iti_split_dir
        self.fold_num = fold_num
        self.data_gen_seed = data_gen_seed
        self.datasets = {'train': {}, 'val': {}, 'test': {}}

        if prompt_template is None:
            self.prompt_template = "Q: {question} A:"
        else:
            self.prompt_template = prompt_template

    ### Load TruthfulQA data from local directory
    def load_data(self, train_size=0):
        for split in self.datasets:
            data_dir = os.path.join(self.iti_split_dir, f'fold_{self.fold_num}_{split}_seed_{self.data_gen_seed}.csv')
            raw_data = self.tqa_formatter_csv_to_hf(
                json.loads(pd.read_csv(open(data_dir, 'r')).to_json(orient='records')))
            ## Only sample part of the training data if train_size !=0
            if train_size != 0 and split == 'train':
                raw_data = random.sample(raw_data, train_size)
            if split == 'train' or split == 'val':
                dpo_dict = self.format_tqa_prompts(raw_data, positive_only=False, preference=True)
                dpo_dict = Dataset.from_dict(dpo_dict)
            else:
                prompts, labels = self.format_tqa_prompts(raw_data, positive_only=False, preference=False)
            self.datasets[split]['preference_data'] = dpo_dict
            self.datasets[split]['hf'] = Dataset.from_pandas(pd.DataFrame(data=self.datasets[split]))
            self.datasets[split]['data_dir'] = data_dir
            self.datasets[split]['raw_data'] = raw_data
        return self.datasets

    ### Convert TruthfulQA data in CSV to Huggingface dataset format
    def tqa_formatter_csv_to_hf(self, csv_dataset):
        hf_data = csv_dataset.copy()
        for i in range(len(hf_data)):
            entry = hf_data[i]
            hf_data[i]['question'] = entry['Question']
            hf_data[i]['mc1_targets'] = {'choices': [entry['Best Answer']] + entry['Incorrect Answers'].split('; '),
                                         'labels': [1] + [0] * len(entry['Incorrect Answers'].split('; '))}
            hf_data[i]['mc2_targets'] = {'choices': [entry['Best Answer']] + entry['Incorrect Answers'].split('; '),
                                         'labels': [1] + [0] * len(entry['Incorrect Answers'].split('; '))}
        return hf_data

    ### Convert TruthfulQA data to preference data format for DPO
    def format_tqa_prompts(self, hf_data, key='mc2_targets', positive_only=False, preference=True, prefix=''):
        prompts = []
        labels = []
        entry_dict = {'prompt': [], 'chosen': [], 'rejected': []}
        for entry in hf_data:
            prompt = self.prompt_template.format(question=entry['question'])
            entry_chosen = []
            entry_rejected = []
            for i in range(len(entry[key]['choices'])):
                label = entry[key]['labels'][i]
                prefix = prefix
                ### If prefereces == True, Format data as preference data and store them in separate lists
                if (not positive_only) and preference:
                    ### Add positive examples to the chosen examples
                    if label == 1:
                        entry_chosen.append(entry[key]['choices'][i])
                    ### Add negative examples to the rejected examples
                    else:
                        entry_rejected.append(entry[key]['choices'][i])
                ### If prefereces == False, do not format data as preference data; instead, store all choices in the same list
                if not preference:
                    choice = entry[key]['choices'][i]
                    prompt = self.prompt_template.format(question=entry['question'])
                    prompt = f"{prompt} {prefix} {choice}"
                    prompts.append(prompt)
                    labels.append(label)
            if preference:
                if len(entry_chosen) != len(entry_rejected):
                    entry_chosen = entry_chosen[:min(len(entry_rejected), len(entry_chosen))]
                    entry_rejected = entry_rejected[:len(entry_chosen)]
                prompt = [prompt for _ in range(len(entry_chosen))]
                entry_dict['prompt'].extend(prompt)
                entry_dict['chosen'].extend(entry_chosen)
                entry_dict['rejected'].extend(entry_rejected)

        if not preference:
            return prompts, labels
        else:
            return entry_dict
