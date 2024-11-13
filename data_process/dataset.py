import torch
from torch.utils.data import Dataset
import src.config as config

class BERTDataset_preprocess(Dataset):
    def __init__(self, data, aug_data = [], tokenizer = None):
        self.data = data
        self.aug_data = aug_data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data) + len(self.aug_data)

    def __getitem__(self, idx):
        
        if idx < len(self.data):
            return self.prepare_train_features(self.data[idx], self.tokenizer), idx
        else:
            return self.prepare_train_features(self.aug_data[idx - len(self.data)], self.tokenizer), idx

    def prepare_train_features(self, examples, tokenizer):
    
        tmp = None
        if isinstance(examples, list):
            tmp = {'context':[], 'question':[]}
            for example in examples:
                tmp["context"].append(example[0])
                tmp["question"].append(example[1])
            examples = tmp
        
        tokenized_examples = tokenizer(
            examples["context" if config.pad_on_right else "question"],
            examples["question" if config.pad_on_right else "context"],
            truncation="only_first" if config.pad_on_right else "only_second",
            max_length=config.max_length,
            stride=config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples["offset_mapping"]

        if tmp is None:

            # Let's label examples!

            for i, offsets in enumerate(offset_mapping):

                input_ids = tokenized_examples["input_ids"][i]

                # e.g. [None, 0, 0, 0, 0, 0, 0, None, 1, None, None, None,......]
                sequence_ids = tokenized_examples.sequence_ids(i)

                token_start_index = 0
                
                while sequence_ids[token_start_index] != (0 if config.pad_on_right else 1):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (0 if config.pad_on_right else 1):
                    token_end_index -= 1

                if len(examples['answer']) > 0: 
                    tokenized_examples['gt'] = [0 for _ in range(config.max_length)]
                    for idx, token_range in enumerate(offsets[token_start_index : token_end_index + 1]):
                        for i in range(token_range[0], token_range[1]):
                            if examples['answer'][i] == 'B':
                                tokenized_examples['gt'][idx + 1] = 2
                            elif examples['answer'][i] == 'I' and tokenized_examples['gt'][idx + 1] == 0:
                                tokenized_examples['gt'][idx + 1] = 1
                    for i in range(token_end_index + 1, config.max_length):
                        tokenized_examples['gt'][i] = config.ignore_index # add -100 to ignore token恩

        tokenized_examples['p_name'] = examples['context'] # to get real p_name
        return tokenized_examples

    def collate_fn(batch):
        
        input_ids = []
        token_type_ids = []
        attention_mask = []
        offset_mapping = []
        p_name = [] # to get real p_name
        gt = []
        index = []

        for each_data, idx in batch:

            input_ids.append(each_data['input_ids'][0])
            token_type_ids.append(each_data['token_type_ids'][0])
            attention_mask.append(each_data['attention_mask'][0])
            offset_mapping.append(each_data['offset_mapping'][0])
            p_name.append(each_data['p_name'])

            index.append(idx)
            if 'gt' in each_data.keys():
                gt.append(each_data['gt'])
        
        return torch.tensor([input_ids, token_type_ids, attention_mask]), index, torch.LongTensor(gt), offset_mapping, p_name
    