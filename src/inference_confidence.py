import os
import sys
import torch
from src import config
from tqdm import tqdm

sys.path.append(os.path.abspath('../'))

def inference(inference_loader, model, tokenizer, all_attribute):
    model.eval()
    with torch.no_grad():

        result_text = []
        result_pos = []
        result_confidence = []

        example_idx = 0
        for batch_data, _, _, offset_mapping, p_names in tqdm(inference_loader):
            # batch_data.shape = ((input_ids, attention_mask, token_type_ids), num_of_p_names * 16, max_length) 
            # 1 p_name has 16 question

            num_of_atts = len(all_attribute)
            batch_logits = model(
                input_ids=batch_data[0].to(config.device),
                attention_mask=batch_data[2].to(config.device),
                token_type_ids=batch_data[1].to(config.device)
            )
            
            # Ensure logits have a batch dimension
            if batch_logits.ndim < 3:
                batch_logits = batch_logits.unsqueeze(0)

            batch_predictions = torch.argmax(batch_logits, dim=-1)

            # Process each input
            for input_ids, token_type_ids, model_prediction, offset, logits, p_name in zip(batch_data[0], batch_data[1], batch_predictions, offset_mapping, batch_logits, p_names):

                # Extract the question from input tokens
                question_start = int(token_type_ids.argmax(-1))
                question = tokenizer.convert_ids_to_tokens(input_ids[question_start : question_start + token_type_ids.sum() - 1])
                question = "".join(question)

                # Create a context string by aligning tokens with offset mappings
                offset = offset[1:question_start - 1]
                context = [' '] * offset[-1][1]  # Last word's end position

                for idx, token in enumerate(tokenizer.convert_ids_to_tokens(input_ids[1:question_start - 1])):
                    if token == '[UNK]':
                        context[offset[idx][0]:offset[idx][1]] = list(p_name[offset[idx][0]:offset[idx][1]])
                    else:
                        context[offset[idx][0]:offset[idx][1]] = list(token.replace('##', ''))
                context = "".join(context)

                # Add a new entry for each attribute
                if example_idx % num_of_atts == 0:
                    result_text.append([context, {}])
                    result_pos.append([context, {}])
                    result_confidence.append([context, {}])

                # Check if result indices align correctly
                if not (example_idx // num_of_atts) + 1 == len(result_pos):
                    print('---------------------------')
                    print(context)
                    print(list(result_pos.keys())[-1])
                    print(example_idx)

                # Confidence calculation
                confidence_scores = []
                last_idx = 0
                for idx, confidence in enumerate(torch.nn.functional.softmax(logits[1:question_start - 1], dim=-1).max(-1)[0].cpu().numpy().tolist()):
                    if last_idx < offset[idx][0]:
                        confidence_scores += [0.0] * (offset[idx][0] - last_idx)
                    confidence_scores += ([0.0] * (offset[idx][1] - offset[idx][0] - 1)) + [confidence]
                    last_idx = offset[idx][1]
                
                result_confidence[-1][1][question] = confidence_scores

                # Process predictions and extract attributes and their positions
                model_prediction = model_prediction[1:]
                start = 0
                attribute_values = []
                attribute_positions = []

                i = 0
                while i <= question_start - 2:
                    if model_prediction[i] == 2:  # End of attribute
                        if i > start:
                            attribute_values.append(context[offset[start][0]:offset[i - 1][1]])
                            attribute_positions.append([offset[start][0], offset[i - 1][1] - 1])
                        start = i
                    elif model_prediction[i] == 1:  # Continue attribute
                        pass
                    elif model_prediction[i] == 0:  # New attribute
                        if i > start:
                            attribute_values.append(context[offset[start][0]:offset[i - 1][1]])
                            attribute_positions.append([offset[start][0], offset[i - 1][1] - 1])
                        start = i + 1
                    i += 1

                result_text[-1][1][question] = attribute_values
                result_pos[-1][1][question] = attribute_positions

                example_idx += 1

        return result_text, result_pos, result_confidence
