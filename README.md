# Sentence Transformers Chinese Version

本项目用来训练中文版的Sentence Transformer模型。

本项目基于UKPLab的[sentence-transformers](https://github.com/UKPLab/sentence-transformers) 实现。

Sentence Transformer是句子编码器，用孪生网络和开放领域的句子匹配数据集训练。

- 基于CMNLI训练
  - 训练方式：在examples/training_transformers下运行python training_cmnli.py
- 基于CSTS训练
  - 训练方式：在examples/training_transformers下运行python training_csts.py
- 基于CQQP训练
  - 训练方式：在examples/training_transformers下运行python training_cqqp.py

