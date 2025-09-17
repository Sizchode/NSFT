import pandas as pd

from .presets import preset_map


def load_questions(filename='questions.csv'):

    """Loads csv of questions into a pandas dataframe"""

    questions = pd.read_csv(filename)
    questions.dropna(axis=1, how='all', inplace=True)  # drop all-null columns

    return questions


def save_questions(questions, filename='answers.csv'):

    """Saves dataframe of questions (with model answers) to csv"""

    questions.to_csv(filename, index=False)


def format_prompt(ser, preset='qa', format='general'):

    """Returns fully formatted prompt (preset + question)"""

    if preset == 'my_null':
        # prompt = 'Q: ' + ser['Question'] + '\n\nA:'
        prompt = '[INST] Q: ' + ser['Question'] + '[/INST]A:'
        return prompt
    if preset == 'null':
        prompt = 'Q: ' + ser['Question'] + '\n\nA:'
        return prompt

    if preset in ['chat', 'long', 'harm']:
        prompt = preset_map[preset].format(ser['Question'])
        return prompt

    if format == 'T5':  # no preset, just the question
        prompt = ser['Question']
        return prompt

    if format == 'UQA':  # no preset, just the question (lowercase)
        prompt = ser['Question'].lower()
        return prompt

    # prompt = ''.join([preset_map[preset], '\n\nQ: ', ser['Question'], '\nA: '])
    prompt = ''.join([preset_map[preset], '\n\nQ: ', ser['Question']])
    return prompt


def format_prompt_with_answer_strings(question, ans, preset='qa', format='general'):

    """Returns fully formatted prompt with answer (preset + question + answer choice)"""

    if preset == 'null':
        prompt = 'Q: ' + question + '\n\nA: ' + ans
        return prompt
    if preset == 'my_null':
        prompt = '[INST] Q: ' + question + '[/INST]A:' + ans
        return prompt
    if preset in ['chat', 'long', 'harm']:
        prompt = preset_map[preset].format(question) + ' ' + ans
        return prompt

    if format == 'T5':
        prompt = question
        return prompt

    prompt = ''.join([preset_map[preset], '\n\nQ: ', question, '\nA: ', ans])
    return prompt



def split_multi_answer(ans, sep=';', close=True):

    """Splits string of all reference answers into a list of formatted answers"""

    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers:
        a = a.strip()
        if len(a):
            if close:  # add a period after all answers
                if a[-1] != '.':
                    split_answers.append(a + '.')
                else:
                    split_answers.append(a)
            else:
                split_answers.append(a)

    return split_answers


def format_best(best_ans, close=True):

    """Formats best answer to match format of reference answers"""

    best = best_ans.strip()
    if close:
        if best[-1] != '.':
            best = best + '.'
    return best


def find_start(token_list):

    """Finds starting index of answer tokens, skipping newlines and prefixes"""

    idx_start = 0

    while token_list[idx_start] == '\n':  # ignore starting newlines
        idx_start += 1

    # if answer starts with 'A:', skip these tokens
    if (token_list[idx_start] == 'A') and (token_list[idx_start + 1] == ':'):
        idx_start += 2

    return idx_start
