import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertForQuestionAnswering, BertPreTrainedModel, BertTokenizer, BertModel


class BertForQuestionAnsweringCustom(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.answerable_outputs = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        answerable=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        
        sequence_output = outputs[0]
        
        #print(outputs[0][0][0].shape,outputs[1][0].shape)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        cls_logit = self.answerable_outputs(sequence_output[:,0]).squeeze(-1)
        outputs = (cls_logit, start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            pos_weight = torch.tensor(0.4)
            bce_loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight)
            answerable_loss = bce_loss_fct(cls_logit, answerable)
            #context_end = []
            #for b in token_type_ids:
            #    for i, pos in enumerate(b):
            #        if pos == 1:
            #            context_end.append(i)
            #            break
            #print(context_end)
            #print(start_positions)
            #print(end_positions)
            #print(total_loss)
            #for i in range(len(end_positions)):
            #    if end_positions[i].item() > context_end[i]:
            #        total_loss[i] = 0
            outputs = (answerable_loss, total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)