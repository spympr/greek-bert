import torch

from torch.utils.data import Dataset

from ...utils.sequences import pad_to_max
from ..utils import parse_ner_dataset_file


class NERBERTDataset(Dataset):
    I2L = [
        'B-FACILITY',
        'B-LOCATION-NAT',
        'B-LOCATION-UNK',
        'B-GPE',
        'B-PERSON',
        'B-PUBLIC-DOCS',
        'B-LEG-REFS',
        'B-ORG',
        'I-FACILITY',
        'I-LOCATION-NAT',
        'I-LOCATION-UNK',
        'I-GPE',
        'I-PERSON',
        'I-PUBLIC-DOCS',
        'I-LEG-REFS',
        'I-ORG',
        'O'
    ]
    L2I = {k: i for i, k in enumerate(I2L)}

    def __init__(self, dataset_file, tokenizer, bert_like_special_tokens, preprocessing_function):

        self.ids = []
        self.texts = []
        self.text_lens = []
        self.pred_masks = []
        self.targets = []

        for i, tokenlist in enumerate(parse_ner_dataset_file(dataset_file)):
            cur_texts, cur_text_lens, pred_mask, labels = self.process_example(
                tokenlist,
                tokenizer,
                bert_like_special_tokens,
                preprocessing_function
            )
            ## Check if we have more than 512 words in a sentence
            if cur_text_lens > 512:
                print(len(cur_texts), i, tokenlist[0])
                # continue

            self.texts.append(cur_texts)
            self.text_lens.append(cur_text_lens)
            self.pred_masks.append(pred_mask)
            self.targets.append([self.L2I.get(cur_l, -1) for cur_l in labels])
            self.ids.append(i)

    def __getitem__(self, index):
        return (
            self.ids[index],
            (self.texts[index], self.text_lens[index]),
            self.targets[index],
            self.pred_masks[index]
        )

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate_fn(batch, pad_value):
        batch_zipped = list(zip(*batch))
        input_zipped = list(zip(*batch_zipped[1]))

        ids = batch_zipped[0]
        texts = torch.tensor(pad_to_max(input_zipped[0], pad_value=pad_value), dtype=torch.long)
        text_lens = torch.tensor(input_zipped[1], dtype=torch.int)
        target = torch.tensor(pad_to_max(batch_zipped[2], pad_value=-1), dtype=torch.long)
        pred_mask = torch.tensor(pad_to_max(batch_zipped[3]), dtype=torch.bool)

        batch = {
            'id': ids,
            'input': [texts, text_lens],
            'target': target,
            'mask': pred_mask
        }

        return batch

    @staticmethod
    def process_example(tokens, tokenizer, bert_like_special_tokens, preprocessing_function):
        transformer_tokens = [tokenizer.cls_token_id] if bert_like_special_tokens else [tokenizer.bos_token_id]
        pred_mask = [0]
        labels = ['PAD']
        for token in tokens:
            processed_token = preprocessing_function(token['text']) if preprocessing_function else token['text']
            current_tokens = tokenizer.encode(processed_token, add_special_tokens=False)
            if len(current_tokens) == 0:
                current_tokens = [tokenizer.unk_token_id]
            transformer_tokens.extend(current_tokens)
            labels.extend([token['label']] + ['PAD'] * (len(current_tokens) - 1))
            pred_mask.extend([1] + [0] * (len(current_tokens) - 1))
        transformer_tokens.append(tokenizer.sep_token_id if bert_like_special_tokens else tokenizer.eos_token_id)
        pred_mask.append(0)
        labels.append('PAD')

        return transformer_tokens, len(transformer_tokens), pred_mask, labels
