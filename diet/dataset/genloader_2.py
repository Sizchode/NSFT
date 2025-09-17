from datasets import load_dataset, Dataset
import random
import pandas as pd
import torch
import numpy as np
from typing import Dict, Any


class LogiQA2:
    def __init__(self, prompt_template=None, chat_template=False, model_name=None, seed=42):
        self.seed = seed
        self.model_name = model_name

        self.prompt_template = prompt_template or (
            "Passage: {context}\n"
            "Question: {question}\n"
            "Options:\n"
            "0. {c0}\n1. {c1}\n2. {c2}\n3. {c3}\n"
            "Answer:\n"
        )
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template} [/INST]\n"
        self.datasets = {}  # 与 OBQA 保持一致

    # ========== 公共接口：与 OBQA 相同 ==========
    def load_data(self, train_size=0):

        ds = load_dataset("datatune/LogiQA2.0")
        split_map = {}
        for s in ["train", "validation", "dev", "test"]:
            if s in ds:
                split_map[s] = ds[s]
        if "validation" not in split_map and "dev" in split_map:
            split_map["validation"] = split_map["dev"]

        if train_size and "train" in split_map:
            split_map["train"] = split_map["train"].shuffle(seed=self.seed).select(range(train_size))

        for split in ["train", "validation", "test"]:
            if split in split_map:
                self.datasets[split] = self.format_prompt(split_map[split])
        return self.datasets

    def format_prompt(self, data):
        formatted = []
        for ex in data:
            context, question, options, ans = self._normalize_one(ex)
            prompt = self.prompt_template.format(
                context=context, question=question,
                c0=options[0], c1=options[1], c2=options[2], c3=options[3]
            )
            gold = str(ans)                # "0".."3"
            text = prompt + gold + "</s>"
            formatted.append({
                "prompt": prompt,
                "true_labels": [gold],
                "text": text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))

    def get_active_set(self, ratio=0.05):
        if "train" not in self.datasets:
            raw = self.load_data(train_size=0)
        total = len(self.datasets["train"])
        k = max(1, int(total * ratio))
        random.seed(self.seed)
        return self.datasets["train"].shuffle(seed=self.seed).select(range(k))

    # ========== 内部：字段规范化 ==========
    def _normalize_one(self, ex):
        """
        兼容 datatune/LogiQA2.0：样本以 JSON 字符串存于 ex['text']。
        若能解析则用解析后的字典；否则按原逻辑处理。
        仍只保留 4 选 1 MRC，输出 (context, question, options[:4], answer_idx in 0..3)。
        """
        # --- 1) 若是 JSON-in-string，先解包 ---
        if isinstance(ex, dict) and "text" in ex and isinstance(ex["text"], str):
            raw = ex["text"].strip()
            if raw.startswith("{") and raw.endswith("}"):
                try:
                    ex = json.loads(raw)
                except Exception:
                    pass  # 解析失败则继续用原 ex

        # --- 2) 取 context / question ---
        context = (
            ex.get("text") or ex.get("context") or ex.get("passage")
            or ex.get("article") or ex.get("document") or ex.get("content")
        )
        if isinstance(context, list):
            context = " ".join(str(x) for x in context)
        if context is None:
            raise ValueError("no context")

        question = (
            ex.get("question") or ex.get("query") or ex.get("q")
            or ex.get("problem") or ex.get("stem")
        )
        if question is None:
            raise ValueError("no question")

        # --- 3) 取 options ---
        options = None
        for k in ("options", "choices", "candidates", "option_list"):
            if k in ex:
                opts = ex[k]
                if isinstance(opts, dict):
                    # 支持 {'A': '...', 'B': '...'} 或 {'0': '...', ...}
                    def _ord_key(s):
                        s = str(s)
                        if s in "ABCD": return "ABCD".index(s)
                        return int(s) if s.isdigit() else 99
                    keys = sorted(opts.keys(), key=_ord_key)
                    options = [str(opts[kk]) for kk in keys][:4]
                elif isinstance(opts, list):
                    options = [str(v) for v in opts][:4]
                break
        if options is None:
            abcd = [ex.get("A"), ex.get("B"), ex.get("C"), ex.get("D")]
            if any(v is not None for v in abcd):
                options = [str(v) for v in abcd if v is not None]
        if not options or len(options) < 4:
            raise ValueError("no 4-way options")
        options = list(options)[:4]

        # --- 4) 取 answer 并映射到 0..3 ---
        ans = (ex.get("answer") or ex.get("label") or ex.get("gold")
            or ex.get("target") or ex.get("answer_idx") or ex.get("correct"))
        if ans is None:
            # 若提供的是正确选项文本
            ans_text = ex.get("answer_text") or ex.get("solution")
            if isinstance(ans_text, str) and ans_text in options:
                ans = options.index(ans_text)

        if isinstance(ans, str):
            s = ans.strip()
            if s in ("A", "B", "C", "D"):
                answer_idx = ord(s) - ord("A")
            elif s.isdigit():
                answer_idx = int(s)
                if answer_idx in (1, 2, 3, 4):  # 容忍 1..4 标注
                    answer_idx -= 1
            elif s in options:
                answer_idx = options.index(s)
            else:
                raise ValueError("bad answer string")
        elif isinstance(ans, int):
            answer_idx = ans - 1 if ans in (1, 2, 3, 4) else ans
        else:
            raise ValueError("bad answer type")

        if not (0 <= answer_idx <= 3):
            raise ValueError("answer out of range")

        return context, question, options, answer_idx



class BoolQ:
    def __init__(self, prompt_template=None, chat_template=False, model_name=None, seed=42):
        self.seed = seed
        self.model_name = model_name
        self.prompt_template = prompt_template or "Q: {question}\nContext:\n{context}\nA:\n"
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template} [/INST]\n"
        self.datasets = {}

    def load_data(self, train_size=0):
        raw_dataset = load_dataset("super_glue", "boolq")
        for split in ['train', 'validation']:
            data = raw_dataset[split]
            if split == 'train' and train_size:
                data = data.shuffle(seed=self.seed).select(range(train_size))
            self.datasets[split] = self.format_prompt(data)
        self.datasets['test'] = self.datasets['validation']
        return self.datasets

    def format_prompt(self, data):
        formatted = []
        for d in data:
            prompt = self.prompt_template.format(
                question=d["question"],
                context=d["passage"]
            )
            answer = "Yes" if d["label"] == 1 else "No"
            text = prompt + answer + '</s>'
            formatted.append({
                'prompt': prompt,
                'true_labels': [answer],
                'text': text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))

    def get_active_set(self, ratio=0.05):
        if 'train' not in self.datasets:
            raw_dataset = load_dataset("super_glue", "boolq")
            train_data = raw_dataset['train']
        else:
            train_data = self.datasets['train']

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)

        if 'train' not in self.datasets:
            train_data = train_data.shuffle(seed=self.seed).select(range(sample_size))
            active_set = self.format_prompt(train_data)
        else:
            active_set = self.datasets['train'].shuffle(seed=self.seed).select(range(sample_size))

        return active_set

    def get_dev_set(self, ratio=0.1, return_rest=True):
        """
        Sample a dev set from BoolQ training data, format like test (prompt + target_text + text),
        return remaining training data if requested.
        """
        raw_dataset = load_dataset("super_glue", "boolq")
        train_data = raw_dataset['train']

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))

        train_data = train_data.shuffle(seed=self.seed)
        dev_data_raw = train_data.select(range(sample_size))
        rest_data_raw = train_data.select(range(sample_size, total_samples))

        # Format dev set
        dev_data = []
        for d in dev_data_raw:
            prompt = self.prompt_template.format(
                question=d["question"],
                context=d["passage"]
            )
            answer = "Yes" if d["label"] == 1 else "No"
            text = prompt + answer + '</s>'
            dev_data.append({
                'prompt': prompt,
                'target_text': answer,
                'true_labels': [answer],  # add here
                'text': text
            })

        dev_dataset = Dataset.from_pandas(pd.DataFrame(dev_data))
        print(dev_dataset.column_names) 
        print(f"\n[BoolQ] Dev set sampled: {sample_size}/{total_samples}")
        for i in range(min(3, sample_size)):
            print(f"\n[Sample {i}]")
            print("Prompt:", dev_data[i]['prompt'])
            print("Label:", dev_data[i]['target_text'])

        if return_rest:
            train_dataset_remainder = self.format_prompt(rest_data_raw)
            return dev_dataset, train_dataset_remainder
        else:
            return dev_dataset

class RTE:
    def __init__(self, prompt_template=None, chat_template=False, model_name=None, seed=42):
        self.seed = seed
        self.model_name = model_name
        self.prompt_template = prompt_template or "Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? A:\n"
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template} [/INST]\n"
        self.datasets = {}

    def load_data(self, train_size=0):
        raw_dataset = load_dataset("super_glue", "rte")
        for split in ['train', 'validation']:
            data = raw_dataset[split]
            if split == 'train' and train_size:
                data = data.shuffle(seed=self.seed).select(range(train_size))
            self.datasets[split] = self.format_prompt(data)
        self.datasets['test'] = self.datasets['validation']
        return self.datasets

    def format_prompt(self, data):
        formatted = []
        for d in data:
            prompt = self.prompt_template.format(
                premise=d["premise"],
                hypothesis=d["hypothesis"]
            )
            answer = "Yes" if d["label"] == 0 else "No"
            text = prompt + answer + '</s>'
            formatted.append({
                'prompt': prompt,
                'true_labels': [answer],
                'text': text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))

    def get_active_set(self, ratio=0.05):
        if 'train' not in self.datasets:
            raw_dataset = load_dataset("super_glue", "rte")
            train_data = raw_dataset['train']
        else:
            train_data = self.datasets['train']

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)

        if 'train' not in self.datasets:
            train_data = train_data.shuffle(seed=self.seed).select(range(sample_size))
            active_set = self.format_prompt(train_data)
        else:
            active_set = self.datasets['train'].shuffle(seed=self.seed).select(range(sample_size))

        return active_set


class HellaSwag:
    def __init__(self, prompt_template=None, chat_template=False, model_name=None, seed=42):
        self.seed = seed
        self.model_name = model_name
        self.prompt_template = prompt_template or "Context: {context}\nQuestion: {question}\nOptions:\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nAnswer:\n"
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template} [/INST]\n"
        self.datasets = {}

    def load_data(self, train_size=0):
        raw_dataset = load_dataset("hellaswag")
        for split in ['train', 'validation']:
            data = raw_dataset[split]
            if split == 'train' and train_size:
                data = data.shuffle(seed=self.seed).select(range(train_size))
            self.datasets[split] = self.format_prompt(data)
        self.datasets['test'] = self.datasets['validation']
        return self.datasets

    def format_prompt(self, data):
        formatted = []
        for d in data:
            options = d["endings"]
            choices = ["A", "B", "C", "D"]
            answer_idx = int(d["label"])

            prompt = self.prompt_template.format(
                context=d["ctx"],
                question="Which is the most likely continuation?",
                option_a=options[0],
                option_b=options[1],
                option_c=options[2],
                option_d=options[3]
            )

            answer = choices[answer_idx]
            text = prompt + answer + '</s>'
            formatted.append({
                'prompt': prompt,
                'true_labels': [answer],
                'text': text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))

    def get_active_set(self, ratio=0.05):
        if 'train' not in self.datasets:
            raw_dataset = load_dataset("hellaswag")
            train_data = raw_dataset['train']
        else:
            train_data = self.datasets['train']

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)

        if 'train' not in self.datasets:
            train_data = train_data.shuffle(seed=self.seed).select(range(sample_size))
            active_set = self.format_prompt(train_data)
        else:
            active_set = self.datasets['train'].shuffle(seed=self.seed).select(range(sample_size))

        return active_set


class WinoGrande:
    def __init__(self, prompt_template=None, chat_template=False, model_name=None, seed=42):
        self.seed = seed
        self.model_name = model_name
        self.prompt_template = prompt_template or "Sentence: {sentence}\nQuestion: {question}\nOptions:\n1. {option1}\n2. {option2}\nAnswer:\n"
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template} [/INST]\n"
        self.datasets = {}

    def load_data(self, train_size=0):
        raw_dataset = load_dataset("winogrande", "winogrande_xl")
        for split in ['train', 'validation']:
            data = raw_dataset[split]
            if split == 'train' and train_size:
                data = data.shuffle(seed=self.seed).select(range(train_size))
            self.datasets[split] = self.format_prompt(data)
        self.datasets['test'] = self.datasets['validation']
        return self.datasets

    def format_prompt(self, data):
        formatted = []
        for d in data:
            sentence = d["sentence"]
            option1 = d["option1"]
            option2 = d["option2"]

            question = f"In the sentence, what does the blank '_' refer to?"

            prompt = self.prompt_template.format(
                sentence=sentence,
                question=question,
                option1=option1,
                option2=option2
            )

            answer = d["answer"]
            text = prompt + answer + '</s>'
            formatted.append({
                'prompt': prompt,
                'true_labels': [answer],
                'text': text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))

    def get_active_set(self, ratio=0.05):
        if 'train' not in self.datasets:
            raw_dataset = load_dataset("winogrande", "winogrande_xl")
            train_data = raw_dataset['train']
        else:
            train_data = self.datasets['train']

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)

        if 'train' not in self.datasets:
            train_data = train_data.shuffle(seed=self.seed).select(range(sample_size))
            active_set = self.format_prompt(train_data)
        else:
            active_set = self.datasets['train'].shuffle(seed=self.seed).select(range(sample_size))

        return active_set


class ARC:
    def __init__(self, subset="easy", prompt_template=None, chat_template=False, model_name=None, seed=42):
        self.seed = seed
        self.model_name = model_name

        if subset.lower() == "easy":
            self.subset = "ARC-Easy"
        elif subset.lower() == "challenge":
            self.subset = "ARC-Challenge"
        else:
            self.subset = subset  # fallback for robustness

        self.prompt_template = prompt_template or "Question: {question}\nOptions:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nE. {E}\nAnswer:\n"
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template} [/INST]\n"
        self.datasets = {}

    def load_data(self, train_size=0):
        raw_dataset = load_dataset("ai2_arc", self.subset)
        for split in ['train', 'validation', 'test']:
            data = raw_dataset[split]
            if split == 'train' and train_size:
                data = data.shuffle(seed=self.seed).select(range(train_size))
            self.datasets[split] = self.format_prompt(data)
        return self.datasets

    def format_prompt(self, data):
        formatted = []
        for d in data:
            choices = {label: text for label, text in zip(d["choices"]["label"], d["choices"]["text"])}
            for opt in ["A", "B", "C", "D", "E"]:
                if opt not in choices:
                    choices[opt] = ""

            prompt = self.prompt_template.format(
                question=d["question"],
                A=choices.get("A", ""),
                B=choices.get("B", ""),
                C=choices.get("C", ""),
                D=choices.get("D", ""),
                E=choices.get("E", "")
            )

            answer = d["answerKey"]
            text = prompt + answer + '</s>'
            formatted.append({
                'prompt': prompt,
                'true_labels': [answer],
                'text': text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))

    def get_active_set(self, ratio=0.05):
        if 'train' not in self.datasets:
            raw_dataset = load_dataset("ai2_arc", self.subset)
            train_data = raw_dataset['train']
        else:
            train_data = self.datasets['train']

        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)

        if 'train' not in self.datasets:
            train_data = train_data.shuffle(seed=self.seed).select(range(sample_size))
            active_set = self.format_prompt(train_data)
        else:
            active_set = self.datasets['train'].shuffle(seed=self.seed).select(range(sample_size))

        return active_set

class OBQA:
    def __init__(self, prompt_template=None, chat_template=False, model_name=None, seed=42):
        self.seed = seed
        self.model_name = model_name
        self.prompt_template = prompt_template or "Question: {question}\nOptions:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:\n"
        if chat_template:
            self.prompt_template = f"<s>[INST] {self.prompt_template} [/INST]\n"
        self.datasets = {}
    def load_data(self, train_size=0):
        raw_dataset = load_dataset("openbookqa", "main")
        for split in ['train', 'validation', 'test']:
            data = raw_dataset[split]
            if split == 'train' and train_size:
                data = data.shuffle(seed=self.seed).select(range(train_size))
            self.datasets[split] = self.format_prompt(data)
        return self.datasets
    def format_prompt(self, data):
        formatted = []
        for d in data:
            # choices = {choice[“label”]: choice[“text”] for choice in d[“choices”]}
            choices = {label: text for label, text in zip(d["choices"]["label"], d["choices"]["text"])}
            prompt = self.prompt_template.format(
                question=d["question_stem"],
                A=choices.get("A", ""),
                B=choices.get("B", ""),
                C=choices.get("C", ""),
                D=choices.get("D", "")
            )
            answer = d["answerKey"]
            text = prompt + answer + '</s>'
            formatted.append({
                'prompt': prompt,
                'true_labels': [answer],
                'text': text
            })
        return Dataset.from_pandas(pd.DataFrame(formatted))
    def get_active_set(self, ratio=0.05):
        if 'train' not in self.datasets:
            raw_dataset = load_dataset("openbookqa", "main")
            train_data = raw_dataset['train']
        else:
            train_data = self.datasets['train']
        total_samples = len(train_data)
        sample_size = max(1, int(total_samples * ratio))
        random.seed(self.seed)
        if 'train' not in self.datasets:
            train_data = train_data.shuffle(seed=self.seed).select(range(sample_size))
            active_set = self.format_prompt(train_data)
        else:
            active_set = self.datasets['train'].shuffle(seed=self.seed).select(range(sample_size))
        return active_set


