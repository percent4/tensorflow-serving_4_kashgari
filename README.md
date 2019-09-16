# tensorflow-serving_4_kashgari
Using tensorflow/serving to deploy kashgari model for time training and predicting.

### 预备工作

需要下载哈工大的预训练模型chinese_wwm_ext.

### 模型训练：

运行model_train.py，训练完后会生成saved_model/time_entity文件。

### 模型部署：

先拉取tensorflow-serving镜像，命令为：

```
docker pull tensorflow/serving
```

再利用tensorflow/serving部署模型，命令为：

```
docker run -t --rm -p 8501:8501 -v "/Users/jclian/PycharmProjects/kashgari_tf_serving/saved_model:/models/" -e MODEL_NAME=time_entity tensorflow/serving
```

### 模型预测

1. 运行runServer.py，启动HTTP服务，用于模型预测
2. 运行predict_test.py代码，输出结果如下：

```
一共预测15个句子。
['9月9日至11日']
['日前', '10月1日', '即日']
['12日', '9月11日']
['9月']
['9月11日']
[]
['近日', '今年2月6日']
['当地时间周四（9月12日）']
['9月12日', '9月2日']
['9月12日下午']
['今天', '目前']
['9月13日']
['2019年6月末', '2019年上半年', '上半年']
['9月11日']
['当日', '2019年']
预测耗时: 15.1085s.
```
