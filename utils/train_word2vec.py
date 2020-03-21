from gensim.models.word2vec import LineSentence
from gensim.models import word2vec

def pratice_word2vec(file_path):
    '''
    训练词向量
    :param file_path: 分词文件路径，str
    :return: 训练好的词向量模型
    '''
    model = word2vec.Word2Vec(LineSentence(file_path), workers=4, min_count=5, size=200)
    return model

def save_word2vec(save_model_path, model):
    '''
    保存词向量到文件
    :param save_model_path: 需要保持的词向量路径,str
    :param model:训练好的词向量
    :return:
    '''
    model.save(save_model_path)

def read_word2vec(model_path):
    '''
    从文件读取训练好的词向量
    :param model_path: 词向量文件路径str
    :return: 词向量模型
    '''
    model = word2vec.Word2Vec.load(model_path)
    return model
