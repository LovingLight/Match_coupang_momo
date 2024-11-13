import config

def prepare_train_features(examples, tokenizer):
    """
    Prepares training features by tokenizing examples and assigning ground truth (gt) labels.
    """
    
    # Handle input format for list of examples
    if isinstance(examples, list):
        examples = {
            'context': [example[0] for example in examples],
            'question': [example[1] for example in examples]
        }

    # Tokenize context and question based on padding configuration
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

    # Remove overflow and offset mapping to manage only required fields
    tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # If examples contain answer labels, assign ground truth labels (gt)
    if 'answer' in examples:
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            sequence_ids = tokenized_examples.sequence_ids(i)

            # Identify the start and end of the context/question in the tokenized sequence
            token_start_index = next(idx for idx, seq_id in enumerate(sequence_ids) 
                                     if seq_id == (0 if config.pad_on_right else 1))
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (0 if config.pad_on_right else 1):
                token_end_index -= 1

            # Initialize ground truth labels
            tokenized_examples['gt'] = [0] * config.max_length

            # Assign BIO labels to the tokenized sequence
            for idx, token_range in enumerate(offsets[token_start_index : token_end_index + 1]):
                for i in range(token_range[0], token_range[1]):
                    if examples['answer'][i] == 'B':
                        tokenized_examples['gt'][idx + 1] = 2  # 'B' label
                    elif examples['answer'][i] == 'I' and tokenized_examples['gt'][idx + 1] == 0:
                        tokenized_examples['gt'][idx + 1] = 1  # 'I' label

            # Set ignored tokens to -100
            for i in range(token_end_index + 1, config.max_length):
                tokenized_examples['gt'][i] = -100

    return tokenized_examples
