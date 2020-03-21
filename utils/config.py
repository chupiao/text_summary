# -*- coding:utf-8 -*-
# Created by LuoJie at 11/16/19
import os
import pathlib

# 预处理数据，构建数据集
is_build_dataset = True

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 训练数据路径
source_data_path = os.path.join(root, 'data', 'news.csv')

# 哈工停用词文件路径
start_stop_words = os.path.join(root, 'data', '哈工大停用词表.txt')

# 分词路径
end_stop_words = os.path.join(root, 'data', 'result', 'cut_words.txt')

# 词频路径
word_frequency = os.path.join(root, 'data', 'result', 'word_frequency.txt')

# 词向量路径
words_model = os.path.join(root, 'data', 'result', 'word2vec.model')