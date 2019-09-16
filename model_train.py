# -*- coding: utf-8 -*-
# time: 2019-09-12
# place: Huangcun Beijing

import kashgari
from kashgari import utils
from kashgari.corpus import DataReader
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model

# 模型训练

train_x, train_y = DataReader().read_conll_format_file('./data/time.train')
valid_x, valid_y = DataReader().read_conll_format_file('./data/time.dev')
test_x, test_y = DataReader().read_conll_format_file('./data/time.test')

bert_embedding = BERTEmbedding('chinese_wwm_ext_L-12_H-768_A-12',
                               task=kashgari.LABELING,
                               sequence_length=128)

model = BiLSTM_CRF_Model(bert_embedding)

model.fit(train_x, train_y, valid_x, valid_y, batch_size=16, epochs=1)

# Save model
utils.convert_to_saved_model(model,
                             model_path='saved_model/time_entity',
                             version=1)