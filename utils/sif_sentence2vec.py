from sklearn.decomposition import PCA
import numpy as np
# 通过SIF算法合成句子向量
def SIF_sentence_embedding(words, word_frequency, word2vec_model, alpha=1e-4):
    '''
    将词向量通过SIF算法合成句子向量
    :param words: 待合成分词, list
    :param word_frequency: 词频， dict
    :param word2vec_model: 词向量模型
    :param alpha: 计算权重的alpha值
    :return: 句子向量
    '''
    max_fre = max(word_frequency.values())
    # words = cut(text, stop_words).split()
    words = [w for w in words if w in word2vec_model]
    if words != []:
        sen_vec = np.zeros_like(word2vec_model.wv[words[0]])
        for w in words:
            fre = word_frequency.get(w, max_fre)
            weight = alpha / (fre + alpha)
            sen_vec += weight * word2vec_model.wv[w]
        # sen_vec /= len(words)
        sen_vec = np.divide(sen_vec, len(words))
        # PCA降噪
        pca = PCA()
        pca.fit(np.array(sen_vec).reshape(-1, 1))
        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT

        if len(u) < len(word2vec_model.wv['测试']):
            for i in range(len(word2vec_model.wv['测试']) - len(u)):
                u = np.append(u, 0)

        sub = np.multiply(u, sen_vec)
        sen_vec = np.subtract(sen_vec, sub)
        return sen_vec
    return ''