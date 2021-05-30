# Sentence Transformers Chinese Version

本项目用来训练中文版的Sentence Transformer模型。

本项目基于UKPLab的[sentence-transformers](https://github.com/UKPLab/sentence-transformers) 实现。

Sentence Transformer是句子编码器，用孪生网络和开放领域的句子匹配数据集训练。

CMNLI、CSTS、CQQP数据下载地址为：oss://tasi-users/zhangyi/projects/sentence_transformers/datasets/，下载后放在examples/datasets下。

- 用CMNLI数据训练：在examples/training_transformers下运行python training_cmnli.py
- 用CSTS数据训练：在examples/training_transformers下运行python training_csts.py
- 用CQQP数据训练：在examples/training_transformers下运行python training_cqqp.py

