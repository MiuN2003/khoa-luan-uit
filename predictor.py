from transformers import AutoTokenizer
from define_model import MRCQuestionAnswering
import torch
from tqdm import tqdm

# Define tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
model = MRCQuestionAnswering.from_pretrained('MiuN2k3/mtl-xlmr-large-dsc-v2', revision='45045206cedc7ed55f14d93ec01e9e8268cb8b8d')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def tokenize_function_2(example):
    # example["question"] = example["question"].split()
    # example["context"] = example["context"].split()
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
            valid = True

    question_sub_words_ids = [[tokenizer.bos_token_id]] + question_sub_words_ids + [[tokenizer.eos_token_id]]
    context_sub_words_ids = context_sub_words_ids + [[tokenizer.eos_token_id]]

    input_ids = [j for i in question_sub_words_ids + context_sub_words_ids for j in i]
    if len(input_ids) > max_len_single_sentence + 2:
        valid = True

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

def data_collator(samples, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(samples) == 0:
        return {}

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

    input_ids = collate_tokens([torch.tensor(item['input_ids']).to(device) for item in samples], pad_idx=tokenizer.pad_token_id)
    attention_mask = torch.zeros_like(input_ids).to(device)
    for i in range(len(samples)):
        attention_mask[i][:len(samples[i]['input_ids'])] = 1
    words_lengths = collate_tokens([torch.tensor(item['words_lengths']).to(device) for item in samples], pad_idx=0)

    batch_samples = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'words_lengths': words_lengths,
    }

    return batch_samples

def extract_answer(inputs, outputs):
    plain_result = []
    for sample_input, start_logit, end_logit, question_logit in zip(inputs, outputs['start_logits'], outputs['end_logits'], outputs['question_logits']):
        sample_words_length = sample_input['words_lengths']
        input_ids = sample_input['input_ids']
        # Get the most likely beginning of answer with the argmax of the score
        answer_start = sum(sample_words_length[:torch.argmax(start_logit)])
        # Get the most likely end of answer with the argmax of the score
        answer_end = sum(sample_words_length[:torch.argmax(end_logit) + 1])

        if answer_start <= answer_end:
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            if answer == tokenizer.bos_token:
                answer = ''
        else:
            answer = ''

        score_start = torch.max(torch.softmax(start_logit, dim=-1)).cpu().detach().numpy().tolist()
        score_end = torch.max(torch.softmax(end_logit, dim=-1)).cpu().detach().numpy().tolist()
        score_label = torch.max(torch.argmax(question_logit[:3], dim=-1)).cpu().detach().numpy().tolist()
        plain_result.append({
            "answer": answer,
            "score_start": score_start,
            "score_end": score_end,
            "score_label": score_label
        })
    return plain_result

def predict(test_dataset):
    # Tokenize the test dataset
    tokenized_test_dataset = test_dataset.map(lambda x: tokenize_function_2(x), batched=False)
    tokenized_test_dataset = tokenized_test_dataset.filter(lambda x: x["valid"])

    # Run the model on the test data
    all_answers = []
    for sample in tqdm(tokenized_test_dataset):
        inputs = [sample]
        inputs_ids = data_collator(inputs,tokenizer)
        with torch.no_grad():
            outputs = model(**inputs_ids)
        answers = extract_answer(inputs, outputs)
        all_answers.extend(answers)
        print(answers)
    return all_answers