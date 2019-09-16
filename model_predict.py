# -*- coding: utf-8 -*-
import time
import requests
from kashgari import utils
import numpy as np

# 从预测的标签中获取相应的预测字段
def get_predict(feature, text, predict_tags):
    times = []

    for i in range(min(len(text), len(predict_tags))):

        time = ''
        if predict_tags[i] == 'B-%s' % feature:
            j = i+1
            time = text[i]
            while j < len(predict_tags) and predict_tags[j] == 'I-%s' % feature:
                time += text[j]
                j += 1

        if time:
            times.append(time)

    return times

def predict():
    t1 = time.time()

    text = '据《新闻联播》报道，9月9日至11日，中央纪委书记赵乐际到河北调研。'
    x = [_ for _ in text]

    # Pre-processor data
    processor = utils.load_processor(model_path='saved_model/time_entity/1')
    tensor = processor.process_x_dataset([x])

    # only for bert Embedding
    tensor = [{
    "Input-Token:0": i.tolist(),
    "Input-Segment:0": np.zeros(i.shape).tolist()
    } for i in tensor]

    # predict
    r = requests.post("http://localhost:8501/v1/models/time_entity:predict", json={"instances": tensor})
    preds = r.json()['predictions']

    # Convert result back to labels
    labels = processor.reverse_numerize_label_sequences(np.array(preds).argmax(-1))

    for label in labels:
        entities = get_predict('TIME', text, label)
        print(entities)

    t2 = time.time()
    print('cost time: %ss.' % str(round(t2 - t1, 4)))

