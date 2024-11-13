import torch
import src.config as config
from transformers import BertPreTrainedModel, BertModel

class Contextual_BERT(BertPreTrainedModel):
    """
    Contextual BERT model for question answering tasks, based on BERT.
    This model integrates the BERT encoder with a question-answering classifier.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.QA = Question_Answering()  # Initialize the question-answering module

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass for the Contextual_BERT model.
        Inputs are passed through the BERT model, and the hidden states are fed into the QA classifier.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs through the BERT model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Extract hidden states from BERT output (batch_size, sequence_length, hidden_dim)
        hidden_states = outputs[0]
        
        # Pass hidden states to the question-answering module
        return self.QA(hidden_states)


class Question_Answering(torch.nn.Module):
    """
    Question Answering (QA) module that applies a BIO classifier on the BERT hidden states.
    The classifier predicts BIO tags (Begin, Inside, Outside) for each token.
    """

    def __init__(self):
        super().__init__()
        self.BIO_classifier = torch.nn.Linear(config.hidden, 3)  # Classifier with 3 output labels (B, I, O)
        self.dropout = torch.nn.Dropout(0.2)  # Dropout for regularization

    def forward(self, batch_hidden_states):
        """
        Forward pass for the QA module.
        Applies dropout and then a linear layer to predict BIO tags.
        """
        # Apply dropout to the hidden states
        batch_hidden_states = self.dropout(batch_hidden_states)
        
        # Predict BIO tags for each token in the sequence
        batch_logits = self.BIO_classifier(batch_hidden_states)  # Shape: (batch_size, sequence_length, 3)
        
        return batch_logits
