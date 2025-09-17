import json
import os
import random

import pandas as pd
from datasets import Dataset


class MQUAKE:
    def __init__(self, split_dir, prompt_template=None, few_shot_template=None, chat_template=False, model_name=None,
                 seed=42):
        self.split_dir = split_dir
        self.datasets = {'train': {}, 'val': {}, 'test': {}}
        self.seed = seed
        if prompt_template is None:
            ### Default prompt template
            self.prompt_template = "Q: Imagine that {edited_prompt}. {question} A:\n"
        else:
            self.prompt_template = prompt_template
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template}[/INST]\n"
        self.few_shot_template = few_shot_template
        self.chat_template = chat_template
        self.model_name = model_name

    def load_data(self, train_size=0):
        for split in self.datasets:
            data_dir = os.path.join(self.split_dir, f'{split}.jsonl')
            raw_data = []
            with open(data_dir, 'r') as f:
                for line in f.readlines():
                    raw_data.append(json.loads(line))
            ### Only sample part of the training data if train_size !=0
            if train_size != 0 and split == 'train':
                raw_data = random.sample(raw_data, train_size)
            formatted_data = self.format_prompt(raw_data)
            self.datasets[split] = formatted_data
        return self.datasets

    def format_prompt(self, data):
        formatted_data = []
        for i in range(len(data)):
            pmeta = data[i]['requested_rewrite'][0]
            true_prompt = pmeta['prompt'].format(pmeta['subject']) + ' ' + pmeta['target_true']['str']
            edited_prompt = pmeta['prompt'].format(pmeta['subject']) + ' ' + pmeta['target_new']['str']
            questions = data[i]['questions']
            new_ans = [data[i]['new_answer']] + data[i]['new_answer_alias']
            concat_prompts = [self.prompt_template.format(edited_prompt=edited_prompt, question=q) for q in questions]
            if self.few_shot_template is not None:
                concat_prompts = [self.few_shot_template + '\n' + q for q in concat_prompts]
            ### Ignore the paraphrases, only use the first prompt + the first correct answer
            if 'gemma' in self.model_name:
                text = concat_prompts[0] + new_ans[0]
            else:
                ### If using llama models, append </s> for better performance
                text = concat_prompts[0] + new_ans[0] + '</s>'
            formatted_data.append({
                'prompt': concat_prompts[0],
                'true_labels': new_ans,
                'text': text
            })
        formatted_data = Dataset.from_pandas(pd.DataFrame(data=formatted_data))
        return formatted_data

    def get_active_set(self, ratio=0.05):
        '''
        Sample a fraction (default 5%) of the training set to form an active set.
        Since the training and test sets in MQUAKE share the same format,
        no additional reformatting is required.
        '''
        if not self.datasets['train']:
            self.load_data(train_size=0)
        train_set = self.datasets['train']
        total_samples = len(train_set)
        sample_size = max(1, int(total_samples * ratio))
        print(sample_size)
        random.seed(self.seed)
        indices = random.sample(range(total_samples), sample_size)
        active_set = train_set.select(indices)
        return active_set

