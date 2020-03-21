import pandas as pd
import jieba
from collections import Counter
import numpy as np

from utils import config as cf
from utils import train_word2vec
from utils.str2float import str2float

def build_data(train_data_path):
    print(1)
    train_df = read_file(train_data_path)
    print(2)
    new_df = train_df.copy(deep=True)
    print(3)
    new_df = split_sentence(new_df, split_text, columns=['content', 'title'])
    print(4)
    stop_words = load_stop_words(cf.start_stop_words)
    print(5)
    main_content = cut_words(train_df, cut, stop_words)
    print(6)
    save_cut_file(cf.end_stop_words, main_content)
    print(7)
    word_frequency = counter_words(main_content)
    print(8)
    save_word_frequency(cf.word_frequency, word_frequency)
    print(9)
    word_model = train_word2vec.pratice_word2vec(cf.end_stop_words)
    print(10)
    train_word2vec.save_word2vec(cf.words_model, word_model)

    print(11)
    return main_content, word_frequency,word_model
def read_file(file_path):
    '''
    加载数据文件
    :param file_path: 文件路径，str
    :return: 加载的数据，dataframe
    '''
    get_data = pd.read_csv(file_path)
    get_data.dropna(subset=['content', 'title'], how='any', inplace=True)
    return get_data

def split_text(text):
    '''
    将文本切割成句子，句子后带标点符号
    :param text: 文本，str
    :return: 句子，list
    '''
    sentences = list()
    # 通过换行符对文档进行分段
    sections = text.split("\n")
    for section in sections:
        if section == "":
            sections.remove(section)

    # 通过分割符对每个段落进行分句
    for i in range(len(sections)):
        section = sections[i]
        text = ""
        for j in range(len(section)):
            char = section[j]
            text = text + char
            if char in ['。', '……', '，', ',', '？', '：', '（', '）', '“', '”', '！', '——'] or j == len(section)-1:
                text = text.strip()
                # 将处理结果加入self.sentences
                sentences.append(text)
                text = ""
    sentences_list = list()
    for i in range(len(sentences)):
        if sentences[i] != "":
            sentences_list.append(sentences[i].strip())
    return sentences_list

def split_sentence(train_df, split_text, columns):
    '''
    分割dataframe中columns列句子
    :param train_df: 需要切割的文本，dataframe
    :param split_text: 切割函数，function
    :param columns: 需要切割的列，list
    :return: 切割后的数据，dataframe
    '''
    # ['content', 'title']
    for col in columns:
        train_df[col] = train_df[col].apply(split_text)
    return train_df

def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径，str
    :return: 停用词表，list
    '''
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    file.close()
    return stop_words

def filter_stopwords(words, stop_words):
    '''
    过滤停用词
    :param words: 切好词的列表, list
    :param stop_words: 加载停用词表后的停用词列表, list
    :return: 过滤后的停用词, list
    '''
    return [word for word in words if word not in stop_words]

def cut(text, stop_words):
    '''
    分词并且过滤停用词
    :param text: 待分词文本，str
    :param stop_words: 停用词列表，list
    :return: 分词且过滤后词，以空格分开，str
    '''
    cut_words = list(jieba.cut(text))
    words = filter_stopwords(cut_words, stop_words)
    return ' '.join(words)

def cut_words(train_data, cut, stop_words):
    '''
    对dataframe中content列分词，并另存到main_content中
    :param train_data: 待分词的文本集，dataframe
    :param cut: 分词过滤停用词函数,function
    :param stop_words: 停用词表, list
    :return: 对content列分词后的新dataframe数据，dataframe
    '''
    main_content = pd.DataFrame()
    main_content['title'] = train_data['title']
    main_content['content'] = train_data['content'].fillna('')
    main_content['tokenized_content'] = main_content['content'].apply(cut, args=(stop_words,))
    return main_content

def save_cut_file(file_path, main_content):
    '''
    保存分词到txt文件
    :param main_content: 带分词的dataframe数据,dataframe
    :return:
    '''
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(' '.join(main_content['tokenized_content'].tolist()))

def counter_words(main_content):
    '''
    统计词频
    :param main_content: 带分词的dataframe数据,dataframe
    :return: 词频
    '''
    tokens = [token for line in main_content['tokenized_content'].tolist() for token in line.split()]
    token_counter = Counter(tokens)
    word_frequency = {w:counts/len(tokens) for w,counts in token_counter.items()}
    return word_frequency

def save_word_frequency(file_path, word_frequency):
    '''
    保持词频到文件
    :param file_path: 文件路径，str
    :param word_frequency: 词频，dict
    :return:
    '''
    with open(file_path, 'w', encoding='utf-8') as f:
        for key, value in word_frequency.items():
            get_str = key + ':' + str(value) + '\n'
            f.write(get_str)

def read_word_frequency(file_path):
    lines = np.loadtxt(file_path, dtype=str, delimiter='\t', encoding='utf-8')
    word_frequency = dict()
    for frequency_list in lines:
        if ':' in frequency_list and frequency_list[-1] not in ['-', 'e']:
            frequency_lists = frequency_list.split(':')
            if len(frequency_lists[0]) < 20:
                word_frequency[frequency_lists[0]] = float(str2float(frequency_lists[1]))
    #word_frequency = {frequency_list.split(':')[0]:frequency_list.split(':')[1] for frequency_list in lines}
    return word_frequency




if __name__ == '__main__':
    #file_path = '../data/zhwiki-latest-pages-articles.xml'
    main_content, word_frequency, word_model = build_data(cf.source_data_path)
    print(main_content['content'].head())
    print(word_frequency['测试'])
    #word_frequency = read_word_frequency(cf.word_frequency)
    #print(word_frequency)
    #word_model = train_word2vec.read_word2vec(cf.words_model)
    print(word_model.wv['帅哥'])