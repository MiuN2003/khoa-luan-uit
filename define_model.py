from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaConfig
from torch import nn
import torch
from torch.nn import CrossEntropyLoss
import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class MRCQuestionAnswering(RobertaPreTrainedModel):
    config_class = RobertaConfig

    def _reorder_cache(self, past, beam_idx):
        pass

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.fc_question_outputs = nn.Linear(config.hidden_size, 1)  # Assuming binary classification for question
        self.label_outputs = nn.Linear(3,1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            words_lengths=None,
            start_idx=None,
            end_idx=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            span_answer_ids=None,
            labels=None,  # New input for question classification labels
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        question_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for question classification.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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

        sequence_output = outputs[0]

        context_embedding = sequence_output

        # Compute align word sub_word matrix
        batch_size = input_ids.shape[0]
        max_sub_word = input_ids.shape[1]
        max_word = words_lengths.shape[1]
        align_matrix = torch.zeros((batch_size, max_word, max_sub_word))

        for i, sample_length in enumerate(words_lengths):
            for j in range(len(sample_length)):
                start_idx = torch.sum(sample_length[:j])
                align_matrix[i][j][start_idx: start_idx + sample_length[j]] = 1 if sample_length[j] > 0 else 0

        align_matrix = align_matrix.to(context_embedding.device)
        # Combine sub_word features to make word feature
        context_embedding_align = torch.bmm(align_matrix, context_embedding)

        logits = self.qa_outputs(context_embedding_align)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        question_logits = self.fc_question_outputs(context_embedding_align).squeeze(-1).contiguous()
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if labels is not None:
                question_loss_fct = CrossEntropyLoss()
                question_logits_combined = torch.cat((question_logits.unsqueeze(-1), start_logits.unsqueeze(-1), end_logits.unsqueeze(-1)),dim=-1)
                # question_logits_mean = torch.mean(question_logits_combined, dim=1)
                question_logits_final = self.label_outputs(question_logits_combined).squeeze(-1).contiguous()
                question_loss = loss_fct(question_logits, labels)
                # print(total_loss, question_loss)
                total_loss = (total_loss + question_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits, question_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return {
            'loss':total_loss,
            'start_logits':start_logits,
            'end_logits':end_logits,
            'hidden_states':outputs.hidden_states,
            'attentions':outputs.attentions,
            'question_logits':question_logits,  # Include question logits in the output
        }
