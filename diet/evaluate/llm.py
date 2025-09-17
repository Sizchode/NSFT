import json
import os

import torch

from .honest_llama.utils import alt_tqa_evaluate


# def evaluate_mquake(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=32,
#                     apply_chat_template=False):
#     results_dir = os.path.join(fname, 'outputs.json')
#     results_json = []
#     tokenizer.padding_side = 'left'
#     inputs = eval_dataset['prompt']

#     iterator = range(0, len(inputs), batch_size)
#     generated = []
#     with torch.no_grad():
#         for i in iterator:
#             inputs_b = inputs[i:i + batch_size]
#             inputs_b = tokenizer(inputs_b, return_tensors='pt', padding=True)
#             inputs_b = {k: v.to(model.device) for (k, v) in inputs_b.items()}
#             outputs = model.generate(**inputs_b, max_new_tokens=max_new_tokens, do_sample=False)
#             decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#             generated.extend(decoded_outputs)


#     corr = 0
#     for i in range(len(generated)):
#         seq = generated[i]
#         if apply_chat_template:
#             sep = '[/INST]\n'
#         else:
#             sep = 'A:\n'
#         ### Extract answer from the generated output
#         ans = seq.split(sep)[-1].split('\n')[0].lower()
#         if '</s>' in ans:
#             ans = ans.replace('</s>', '')
#         entry_corr = 0
#         ### Check if the answer is in the list of correct answers
#         if ans in [label.lower() for label in eval_dataset['true_labels'][i]]:
#             entry_corr = 1
#         corr += entry_corr
#         result = {'prompt': inputs[i], 'response': seq, 'pred': ans, 'label': eval_dataset['true_labels'][i],
#                   'correct': entry_corr}
#         results_json.append(result)
#     accuracy = corr / len(generated)
#     print(f"Accuracy: {accuracy}")
#     json.dump(results_json, open(results_dir, 'w'))
#     return accuracy, generated
import os
import json
import re
import torch

def evaluate_mquake(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=16,
                    apply_chat_template=False):
    results_dir = os.path.join(fname, 'outputs.json')
    results_json = []
    tokenizer.padding_side = 'left'
    inputs = eval_dataset['prompt']

    iterator = range(0, len(inputs), batch_size)
    generated = []

    with torch.no_grad():
        for i in iterator:
            inputs_b = inputs[i:i + batch_size]
            inputs_b = tokenizer(inputs_b, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs_b = {k: v.to(model.device) for (k, v) in inputs_b.items()}
            outputs = model.generate(**inputs_b, max_new_tokens=max_new_tokens, do_sample=False)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded_outputs)

    # def parse_answer(output):
    #     if 'A:' in output:
    #         answer = output.split('A:')[-1]
    #     else:
    #         answer = output
    #     answer = answer.split('</s>')[0]
    #     answer = answer.strip().lower()
    #     answer = re.sub(r'\s+', ' ', answer)
    #     return answer
    def parse_answer(output):
        if 'A:' in output:
            answer = output.split('A:')[-1]
        else:
            answer = output

        answer = re.split(r'\b[B-D]:', answer)[0]
        answer = answer.split('</s>')[0]

        lines = [l.strip() for l in answer.strip().split('\n') if l.strip()]
        if lines:
            answer = lines[0]
        else:
            answer = ""

        # 标准化
        answer = answer.lower()
        answer = re.sub(r'\s+', ' ', answer)
        answer = re.sub(r'[^\w\s]', '', answer)
        return answer

    corr = 0
    for i in range(len(generated)):
        seq = generated[i]
        ans = parse_answer(seq)
        label_set = [label.lower().strip() for label in eval_dataset['true_labels'][i]]
        entry_corr = int(ans in label_set)
        corr += entry_corr

        result = {
            'prompt': inputs[i],
            'response': seq,
            'pred': ans,
            'label': eval_dataset['true_labels'][i],
            'correct': entry_corr
        }
        results_json.append(result)

    accuracy = corr / len(generated)
    print(f"Accuracy: {accuracy:.4f}")
    json.dump(results_json, open(results_dir, 'w'), indent=2)
    return accuracy, generated


# def evaluate_clutrr(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=16,
#                     apply_chat_template=False):
#     results_dir = os.path.join(fname, 'outputs.json')
#     result_json = []
#     tokenizer.padding_side = 'left'

#     inputs = eval_dataset['prompt']
#     def normalize_answer(ans):
#         ans = ans.lower().strip()
#         ans = ans.replace('</s>', '').replace('<pad>', '').replace('<|endoftext|>', '') 
#         ans = re.sub(r'\s+', ' ', ans)      
#         ans = ans.split('-')[0]             
#         ans = ans.split()[0]                
#         return ans
#     iterator = range(0, len(inputs), batch_size)
#     generated = []
#     with torch.no_grad():
#         for i in iterator:
#             inputs_b = inputs[i:i + batch_size]
#             inputs_b = tokenizer(inputs_b, return_tensors='pt', padding=True)
#             inputs_b = {k: v.to(model.device) for (k, v) in inputs_b.items()}
#             outputs = model.generate(**inputs_b, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False,)# use_cache=False to avoid .float issue in Hybrid Cache
#             decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#             generated.extend(decoded_outputs)

#     corr = 0
#     for i in range(len(generated)):

#         seq = generated[i]
#         if apply_chat_template:
#             sep = '[/INST]\n'
#         else:
#             sep = "\'s\n"
#         ans = seq.split(sep)[1].split('.')[0].strip().lower()
#         if '</s>' in ans:
#             ans = ans.replace('</s>', '')
#         gold_label = eval_dataset['target_text'][i].lower()
#         entry_corr = 0
#         if ans == gold_label:
#             entry_corr = 1
#         corr += entry_corr
#         result = {'prompt': inputs[i], 'response': seq, 'pred': ans, 'label': gold_label, 'correct': entry_corr}
#         result_json.append(result)
#     print('Accuracy:', corr / len(generated))
#     json.dump(result_json, open(results_dir, 'w'))
#     return corr / len(generated), generated
def evaluate_clutrr(eval_dataset, model_name, model, tokenizer, fname, batch_size=16, max_new_tokens=16,
                    apply_chat_template=False):  
    import os, json, torch, re

    results_dir = os.path.join(fname, 'outputs.json')
    result_json = []
    tokenizer.padding_side = 'left'
    inputs = eval_dataset['prompt']
    iterator = range(0, len(inputs), batch_size)
    generated = []

    def normalize_prediction(ans: str) -> str:
        """
        将模型输出 `ans` 规范化为 18 个亲属关系标签之一。
        仅修改解析逻辑；推理、文件写入等流程保持不变。
        """
        import re

        # 1) 通用清洗
        ans = ans.lower()
        ans = re.sub(r"</s>|<pad>|<\|endoftext\|>|[<>]", " ", ans)
        ans = re.sub(r"\s+", " ", ans).strip()

        # 2) 去除连续重复（如 'grandfathergrandfather'）
        ans = re.sub(
            r"(father-in-law|mother-in-law|grandmother|grandfather|granddaughter|grandson|"
            r"father|mother|daughter|son|brother|sister|uncle|aunt|nephew|niece)"
            r"\1+",
            r"\1",
            ans,
        )

        # 3) 正则抽取首个合法关系词
        rel_pattern = re.compile(
            r"\b("
            r"father-in-law|mother-in-law|granddaughter|grandson|"
            r"grandmother|grandfather|daughter|mother|brother|sister|father|"
            r"nephew|niece|uncle|aunt|son"
            r")\b"
        )
        match = rel_pattern.search(ans)
        if match:
            return match.group(1)

        # 4) 兜底：返回首词，避免空串
        return ans.split()[0] if ans else ""

    with torch.no_grad():
        for i in iterator:
            inputs_b = inputs[i:i + batch_size]
            inputs_b = tokenizer(inputs_b, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs_b = {k: v.to(model.device) for (k, v) in inputs_b.items()}
            outputs = model.generate(**inputs_b, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded_outputs)

    corr = 0
    for i in range(len(generated)):
        seq = generated[i]

        sep = '[/INST]\n' if apply_chat_template else "'s\n"
        try:
            ans_raw = seq.split(sep)[1].split('.')[0]
        except IndexError:
            ans_raw = seq

        pred = normalize_prediction(ans_raw)
        label = eval_dataset['target_text'][i].lower().strip()  
        entry_corr = int(pred == label)
        corr += entry_corr

        result_json.append({
            'prompt': inputs[i],
            'response': seq,
            'pred': pred,
            'label': label,
            'correct': entry_corr
        })

    accuracy = corr / len(generated)
    print(f"Accuracy: {accuracy:.4f}")
    with open(results_dir, 'w') as f:
        json.dump(result_json, f, indent=2)

    return accuracy, generated


def evaluate_tqa(eval_dataset, fname, model, tokenizer, metrics, model_name=None, verbose=False):
    ### Create directories to save truthfulqa outputs
    if not os.path.exists('./tqa_results/answer_dump'):
        os.makedirs('./tqa_results/answer_dump')
    if not os.path.exists('./tqa_results/summary_dump'):
        os.makedirs('./tqa_results/summary_dump')
    curr_fold_results = alt_tqa_evaluate(
        {model_name: model},
        metric_names=metrics,
        input_path=eval_dataset['data_dir'],
        output_path=f'./tqa_results/answer_dump/{model_name}_truthfulqa.csv',
        summary_path=f'./tqa_results/summary_dump/{model_name}_truthfulqa.csv',
        device="cuda",
        tokenizer=tokenizer,
        ### Print generated outputs
        verbose=verbose,
        ### Use the standard QA prompt for evaluation
        preset='qa'
    )
    print(curr_fold_results)

