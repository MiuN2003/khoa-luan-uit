import json
from glob import glob
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk import word_tokenize as lib_tokenizer
import datasets
from transformers import AutoTokenizer
import torch
from nltk import word_tokenize

dict_map = dict()

def word_tokenize(text):
    global dict_map
    words = text.split()
    words_norm = []
    for w in words:
        if dict_map.get(w, None) is None:
            dict_map[w] = ' '.join(lib_tokenizer(w)).replace('`', '"').replace("''", '"')
        words_norm.append(dict_map[w])
    return words_norm

def strip_answer_string(text):
    text = text.strip()
    while text[-1] in '.,/><;:\'"[]{}+=-_)(*&^!~':
        if text[0] != '(' and text[-1] == ')' and '(' in text:
            break
        if text[-1] == '"' and text[0] != '"' and text.count('"') > 1:
            break
        text = text[:-1].strip()
    while text[0] in '.,/><;:\'"[]{}+=-_)(*&^!~':
        if text[0] == '"' and text[-1] != '"' and text.count('"') > 1:
            break
        text = text[1:].strip()
    text = text.strip()
    return text

def strip_context(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def handle_file(content, claim, evidence, label):
    norm_samples = []
    context_raw = content
    answer_raw = content
    question = claim
    answer_index_raw = int(0)
    label = 0 

    if len(answer_raw) > 0:
        # if context_raw[answer_index_raw: answer_index_raw + len(answer_raw)] == answer_raw:
            context_prev = strip_context(context_raw[:answer_index_raw])
            answer = strip_answer_string(answer_raw)
            answer = answer_raw
            context_next = strip_context(context_raw[answer_index_raw + len(answer):])

            context_prev = ' '.join(word_tokenize(context_prev))
            context_next = ' '.join(word_tokenize(context_next))
            answer = ' '.join(word_tokenize(answer))
            question = ' '.join(word_tokenize(question))

            context = "{} {} {}".format(context_prev, answer, context_next).strip()
            
            norm_samples.append({
                "context": context,
                "question": question,
                "answer_text": answer,
                "answer_start_idx": len("{} {}".format(context_prev, answer).strip()) - len(answer),
                "label": label
            })
    else:
        context_raw = word_tokenize(context_raw)
        question = word_tokenize(question)
        norm_samples.append({
            "context": context_raw,
            "question": question,
            "answer_text": '',
            "answer_start_idx": 0,
            "label": label
        })
    return norm_samples

import datasets
from datasets import concatenate_datasets
import glob
import argparse

datasets.disable_progress_bar()

def assert_sample(sample):
    assert sample['context'][sample['answer_start_idx']: sample['answer_start_idx'] + len(sample['answer_text'])] == \
           sample['answer_text'], sample
    assert len(sample['context']) > 0
    assert len(sample['question']) > 0
    assert int(sample['label']) > -1
    return True

def format_sample(sample):
    context_prev = sample['context'][:sample['answer_start_idx']].split()
    sample['answer_word_start_idx'] = len(context_prev)
    sample['answer_word_end_idx'] = len(context_prev) + len(sample['answer_text'].split()) - 1
    return sample


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


def data_collator(samples):
    if len(samples) == 0:
        return {}

    for sample in samples:
        start_idx = sum(sample['words_lengths'][:sample['start_idx']])
        end_idx = sum(sample['words_lengths'][:sample['end_idx'] + 1])
        sample['span_answer_ids'] = sample['input_ids'][start_idx:end_idx]

    def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res

    input_ids = collate_tokens([torch.tensor(item['input_ids']) for item in samples], pad_idx=tokenizer.pad_token_id)
    attention_mask = torch.zeros_like(input_ids)
    for i in range(len(samples)):
        attention_mask[i][:len(samples[i]['input_ids'])] = 1
    words_lengths = collate_tokens([torch.tensor(item['words_lengths']) for item in samples], pad_idx=0)
    answer_start = collate_tokens([torch.tensor([item['start_idx']]) for item in samples], pad_idx=0)
    answer_end = collate_tokens([torch.tensor([item['end_idx']]) for item in samples], pad_idx=0)
    span_answer_ids = collate_tokens([torch.tensor(item['span_answer_ids']) for item in samples], pad_idx=-100)
    labels = torch.tensor([item['label'] for item in samples])
    # print('answer_start', answer_start,'\nlabels', labels)
    batch_samples = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'words_lengths': words_lengths,
        'start_positions': answer_start,
        'end_positions': answer_end,
        'span_answer_ids': span_answer_ids,
        'labels': labels # Include question labels in the batch
    }
    return batch_samples

def tokenize_function(example):
    example["question"] = example["question"].split()
    example["context"] = example["context"].split()
    max_len_single_sentence = 368

    question_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in example["question"]]
    context_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in example["context"]]
    valid = True
    if len([j for i in question_sub_words_ids + context_sub_words_ids for j in
            i]) > max_len_single_sentence - 1:
        question_ids = [j for i in question_sub_words_ids for j in i]
        context_ids = [j for i in context_sub_words_ids[:example['answer_word_end_idx'] + 1] for j in i]
        remain_tokens = max_len_single_sentence - 1 - len(question_ids)
        if len(question_ids + context_ids) < max_len_single_sentence - 1:
            context_sub_words_ids_revise = context_sub_words_ids[:example['answer_word_end_idx'] + 1]
            idx = example['answer_word_end_idx'] + 1
            while len([j for i in (context_sub_words_ids_revise + [context_sub_words_ids[idx]]) for j in
                       i]) < remain_tokens and idx < len(context_sub_words_ids):
                context_sub_words_ids_revise.append(context_sub_words_ids[idx])
                idx += 1
            context_sub_words_ids = context_sub_words_ids_revise
        else:
            valid = False

    question_sub_words_ids = [[tokenizer.bos_token_id]] + question_sub_words_ids + [[tokenizer.eos_token_id]]
    context_sub_words_ids = context_sub_words_ids + [[tokenizer.eos_token_id]]

    input_ids = [j for i in question_sub_words_ids + context_sub_words_ids for j in i]
    if len(input_ids) > max_len_single_sentence + 2:
        valid = False

    words_lengths = [len(item) for item in question_sub_words_ids + context_sub_words_ids]

    return {
        "input_ids": input_ids,
        "words_lengths": words_lengths,
        "start_idx": (example['answer_word_start_idx'] + len(question_sub_words_ids)) if len(
            example["answer_text"]) > 0 else 0,
        "end_idx": (example['answer_word_end_idx'] + len(question_sub_words_ids)) if len(
            example["answer_text"]) > 0 else 0,
        "valid": valid,
        "label": example["label"] 
    }

def get_dataloader(test_path, num_proc=1):

    test_set = datasets.load_from_disk(test_path)
    print("Test set: ", len(test_set))

    test_set = test_set.map(tokenize_function, batched=False, num_proc=num_proc).filter(
        lambda example: example['valid'], num_proc=num_proc)

    print("Test set: ", len(test_set))
    return test_set

def get_dataset(content, claim):
    dict_data_squad = [handle_file(content=content, claim=claim, evidence=content, label=0)]
    json_path = 'data_test/test.json'

    with open(json_path, 'w', encoding='utf-8') as file:
        for item in dict_data_squad:
            file.write("{}\n".format(json.dumps(item, ensure_ascii=False)))
    print("Output path: ", json_path)
    print("Total: {} samples".format(len(dict_data_squad)))

    dataset_path = 'data_test/test.dataset'
    data_set = []

    for part in glob.glob(f'{json_path}'):
        dataset = datasets.load_dataset('json', data_files=[part])['train']
        dataset = dataset.filter(assert_sample)
        dataset = dataset.map(format_sample)
        data_set.append(dataset)

    test_dataset = concatenate_datasets(data_set)
    test_dataset.save_to_disk(dataset_path)

    print("Test: {} samples".format(len(test_dataset)))
    test_dataset = get_dataloader(test_path='data_test/test.dataset')
    return test_dataset