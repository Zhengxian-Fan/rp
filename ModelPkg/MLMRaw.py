import pytorch_pretrained_bert as Bert
import sys
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch
# class BertConfig(Bert.modeling.BertConfig):
#     def __init__(self, config):
#         super(BertConfig, self).__init__(
#             vocab_size_or_config_json_file=config.get('vocab_size'),
#             hidden_size=config['hidden_size'],
#             num_hidden_layers=config.get('num_hidden_layers'),
#             num_attention_heads=config.get('num_attention_heads'),
#             intermediate_size=config.get('intermediate_size'),
#             hidden_act=config.get('hidden_act'),
#             hidden_dropout_prob=config.get('hidden_dropout_prob'),
#             attention_probs_dropout_prob=config.get('attention_probs_dropout_prob'),
#             max_position_embeddings=config.get('max_position_embedding'),
#             initializer_range=config.get('initializer_range'),
#         )
#         self.seg_vocab_size = config.get('seg_vocab_size')
#         self.age_vocab_size = config.get('age_vocab_size')
#         self.year_vocab_size = config.get('year_vocab_size')
#         self.concat_embeddings = False

#         if config.get('concat_embeddings',-1) !=-1:
#             self.concat_embeddings = config.get('concat_embeddings')


    

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, segment, age
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        self.year_embeddings = nn.Embedding(config.year_vocab_size, config.hidden_size)

        self.posi_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size).\
            from_pretrained(embeddings=self._init_posi_embedding(config.max_position_embeddings, config.hidden_size))

        self.LayerNorm = Bert.modeling.BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        if self.config.concat_embeddings:
            self.catmap = nn.Linear(4 *  config.hidden_size, config.hidden_size)
            self.tanh = nn.Tanh()
    def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None, year_ids=None):
        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)

        word_embed = self.word_embeddings(word_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)
        year_embed = self.year_embeddings (year_ids)
        
        if self.config.concat_embeddings:
            embeddings = self.tanh (self.catmap(torch.cat((word_embed,age_embed,posi_embeddings,year_embed), dim=2) ))  + segment_embed

        else:
            embeddings = word_embed + segment_embed + age_embed + posi_embeddings+year_embed
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def _init_posi_embedding(self, max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos/(10000**(2*idx/hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos/(10000**(2*idx/hidden_size)))

        # initialize position embedding table
        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)

        # reset table parameters with hard encoding
        # set even dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        # set odd dimension
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)

        return torch.tensor(lookup_table)
class BertModel(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = Bert.modeling.BertEncoder(config=config)
        self.pooler = Bert.modeling.BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, year_ids = None,  attention_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(input_ids)
        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, age_ids, seg_ids, posi_ids, year_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output
class BertForMaskedLM(Bert.modeling.BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = Bert.modeling.BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, year_ids = None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, age_ids ,seg_ids, posi_ids, year_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss, prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)
        else:
            return prediction_scores