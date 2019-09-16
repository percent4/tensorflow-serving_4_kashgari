# -*- coding: utf-8 -*-
import requests
from kashgari import utils
import numpy as np
from model_predict import get_predict

import json
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
import traceback

# tornado高并发
import tornado.web
import tornado.gen
import tornado.concurrent
from concurrent.futures import ThreadPoolExecutor

# 定义端口为12333
define("port", default=16016, help="run on the given port", type=int)

# 模型预测
class ModelPredictHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(max_workers=5)

    # get 函数
    @tornado.gen.coroutine
    def get(self):
        origin_text = self.get_argument('text')
        result = yield self.function(origin_text)
        self.write(json.dumps(result, ensure_ascii=False))

    @tornado.concurrent.run_on_executor
    def function(self, text):
        try:
            text = text.replace(' ', '')
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

            entities = get_predict('TIME', text, labels[0])

            return entities

        except Exception:
            self.write(traceback.format_exc().replace('\n', '<br>'))


# get请求
class HelloHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('Hello from lmj from Daxing Beijing!')


# 主函数
def main():
    # 开启tornado服务
    tornado.options.parse_command_line()
    # 定义app
    app = tornado.web.Application(
            handlers=[(r'/model_predict', ModelPredictHandler),
                      (r'/hello', HelloHandler),
                      ], #网页路径控制
          )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

main()