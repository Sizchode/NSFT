from datasets import load_dataset, Dataset
import random
import pandas as pd


# -------------------------------------
# MuSiQue Loader (using huggingface dataset: 'musique')
# -------------------------------------
class MuSiQue:
    def __init__(self, prompt_template=None, chat_template=False, model_name=None, seed=42):
        self.seed = seed
        self.model_name = model_name
        self.prompt_template = prompt_template or "Q: {question}\nContext:\n{context}\nA:\n"
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template} [/INST]\n"
        self.datasets = {}

    def load_data(self, train_size=0):
        raw_dataset = load_dataset("dgslibisey/MuSiQue")  # or use "muSiQue" if alias changes
        split_map = {'train': 'train', 'val': 'validation'}
        for split in split_map:
            data = raw_dataset[split_map[split]]
            if split == 'train' and train_size:
                data = data.shuffle(seed=self.seed).select(range(train_size))
            self.datasets[split] = self.format_prompt(data)
        return self.datasets

    def format_prompt(self, data):
        formatted = []
        for d in data:
            prompt = self.prompt_template.format(
                question=d["question"],
                context="\n".join(d["paragraphs"])
            )
            text = prompt + d["answer"] + '</s>'
            formatted.append({
                'prompt': prompt,
                'true_labels': [d["answer"]],
                'text': text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))

    def get_active_set(self, ratio=0.05):
        """
        Sample a fraction (default 5%) of the raw training data and preprocess it.
        Returns a formatted dataset that can be used for active learning.
        """
        if 'train' not in self.datasets:
            raw_dataset = load_dataset("dgslibisey/MuSiQue", "full")
            train_data = raw_dataset['train']
        else:
            raise ValueError("Dataset already loaded. Please use the loaded dataset for sampling.")

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)
        print(f"Total samples: {total_samples}, Sample size: {sample_size}")

        train_data = train_data.shuffle(seed=self.seed).select(range(sample_size))

        active_set = self.format_prompt(train_data)
        return active_set


class GSM8K:
    def __init__(self, prompt_template="Q: {question}\nA:", seed=42):
        self.prompt_template = prompt_template
        self.seed = seed
        self.datasets = {}

    def load_data(self, train_size=0):
        raw_dataset = load_dataset("openai/gsm8k", "main")
        for split in ['train', 'test']:
            data = raw_dataset[split]
            if split == 'train' and train_size:
                data = data.shuffle(seed=self.seed).select(range(train_size))
            self.datasets[split] = self.format_prompt(data)
        return self.datasets

    def format_prompt(self, data):
        formatted = []
        for d in data:
            prompt = self.prompt_template.format(question=d["question"])
            # Extract the final answer after '####'
            answer = d["answer"].split("####")[-1].strip()
            text = f"Q:{prompt} A:\n{answer} </s>"
            formatted.append({
                'prompt': prompt,
                'true_labels': [answer],
                'text': text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))

    def get_active_set(self, ratio=0.05):
        """
        Sample a fraction (default 5%) of the raw training data and preprocess it.
        Returns a formatted dataset that can be used for active learning.
        """
        if 'train' not in self.datasets:
            # Load data if not already loaded
            raw_dataset = load_dataset("openai/gsm8k", "main")
            train_data = raw_dataset['train']
        else:
            # Use already loaded data
            train_data = self.datasets['train']

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)
        print(f"Total samples: {total_samples}, Sample size: {sample_size}")

        # If we're working with the raw dataset, we need to sample and format
        if 'train' not in self.datasets:
            # Shuffle and select samples
            train_data = train_data.shuffle(seed=self.seed).select(range(sample_size))
            # Format the sampled data
            active_set = self.format_prompt(train_data)
        else:
            # If we already have formatted data, just sample from it
            active_set = self.datasets['train'].shuffle(seed=self.seed).select(range(sample_size))

        return active_set