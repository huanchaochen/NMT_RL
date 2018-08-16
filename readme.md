## 数据集
数据文件在/data目录下，包括英文原文和法文译文

## 预处理
运行 preprocess.py 进行数据预处理：

 - 建立一个中文到整数数字的双向转换的字典
 - 同时需要添加以下的几个特殊字符: 
		 - \<PAD\>: 字符补全
		 - \<EOS\>：句子的结束
		 - \<GO\>: 句子的起始
		 - \<UNK\>： 未出现过的词或者低频词

## 预训练
运行 pretrain.py 将数据集划分为训练集，验证集和测试集，并进行NMT模型的预训练

超参数设置：

    # Number of Epochs  
    epochs = 10  
    # Batch Size  
    batch_size = 128  
    # RNN Size  
    rnn_size = 64  
    # Number of Layers  
    num_layers = 2  
    # Embedding Size  
    encoding_embedding_size = 64  
    decoding_embedding_size = 64  
    # Learning Rate  
    learning_rate = 0.001  
    # Dropout Keep Probability  
    keep_probability = 0.8  
    # display step  
    display_step = 100
  
## 强化学习训练
运行 reinforce.py 载入已训练的模型，并进行强化学习训练

超参数设置：

    # Number of Epochs  
    epochs = 5  
    # Batch Size  
    batch_size = 128  
    # Embedding Size  
    encoding_embedding_size = 64  
    decoding_embedding_size = 64  
    # RNN Size  
    rnn_size = 64  
    # Number of Layers  
    num_layers = 2  
    # Learning Rate  
    learning_rate = 0.0001  
    # Dropout Keep Probability  
    keep_probability = 0.8  
    display_step = 10

## 测试
运行 test.py 进行测试，打印结果

## 模型保存
预训练模型：
`util.save_params(save_path, 'params_actor_sup.p')`
actor和critic：
`util.save_params(save_path,'params_actor_reinforce.p')  
util.save_params(save_path_critic, 'params_critic_sup.p')`

## 训练可视化
保存在/img 文件夹下
