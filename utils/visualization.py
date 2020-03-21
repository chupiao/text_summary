from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pylab import mpl
import random

from utils import train_word2vec
from utils import config as cf

def tsns_words(word_model):
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    # 因为词向量文件比较大，全部可视化就什么都看不见了，所以随机抽取一些词可视化
    words = list(word_model.wv.vocab)
    random.shuffle(words)
    print(1)
    vector = word_model.wv[words]
    tsne = TSNE(n_components=2, init='pca', verbose=1)
    embedd = tsne.fit_transform(vector)
    print(2)
    # 可视化
    plt.figure(figsize=(14, 10))
    plt.scatter(embedd[:300, 0], embedd[:300, 1])

    for i in range(300):
        x = embedd[i][0]
        y = embedd[i][1]
        plt.text(x, y, words[i])
    plt.savefig('../data/result/words.png')
    print(3)
    plt.show()

if __name__ == '__main__':
    word_model = train_word2vec.read_word2vec(cf.words_model)
    print(word_model.wv.most_similar(['美女'], topn=10))
    tsns_words(word_model)