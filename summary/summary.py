#coding=UTF-8
from gensim.models import FastText
import jieba
import jieba.analyse
import jieba.posseg
import numpy as np
from scipy.spatial.distance import cosine

from utils.sif_sentence2vec import SIF_sentence_embedding
from utils.data_loader import split_text, cut, load_stop_words
from utils import config as cf
from utils.data_loader import read_word_frequency
from utils.train_word2vec import read_word2vec

# 引入日志配置
import logging

class GetSummary:
    text = ""
    title = ""
    first_summary = "" #第一次摘要转换成字符串
    first_ratio=0.8          #第一次摘要的截取句子数
    text_ratio = 0.1   #余弦相似度计算后，text权重
    title_ratio = 0.9  #余弦相似度计算后，title权重
    smoth_win = 5   #knn平滑窗口
    mun_len = 9    #截取评分后的排名前num_len的句子
    score={}
    corr_score = {} #句子和分数
    keywords = list()
    sentences = list()
    summary = list()

    def SetText(self, title, text):
        self.title = title
        self.text = text

    def __SplitSentence(self):
        # 通过换行符对文档进行分段
        sections = self.text.split("\n")
        for section in sections:
            if section == "":
                sections.remove(section)

        # 通过分割符对每个段落进行分句
        for i in range(len(sections)):
            section = sections[i]
            text = ""
            k = 0
            for j in range(len(section)):
                char = section[j]
                text = text + char
                if char in ["!",  "。", "？"] or j == len(section)-1:
                    text = text.strip()
                    sentence = dict()
                    sentence["text"] = text
                    sentence["pos"] = dict()
                    sentence["pos"]["x"] = i
                    sentence["pos"]["y"] = k
                    # 将处理结果加入self.sentences
                    self.sentences.append(sentence)
                    text = ""
                    k = k + 1

        for sentence in self.sentences:
            sentence["text"] = sentence["text"].strip()
            if sentence["text"] == "":
                self.sentences.remove(sentence)

        # 对文章位置进行标注，通过mark列表，标注出是否是第一段、尾段、第一句、最后一句
        lastpos = dict()
        lastpos["x"] = 0
        lastpos["y"] = 0
        lastpos["mark"] = list()
        for sentence in self.sentences:
            pos = sentence["pos"]
            pos["mark"] = list()
            if pos["x"] == 0:
                pos["mark"].append("FIRSTSECTION")
            if pos["y"] == 0:
                pos["mark"].append("FIRSTSENTENCE")
                lastpos["mark"].append("LASTSENTENCE")
            if pos["x"] == self.sentences[len(self.sentences)-1]["pos"]["x"]:
                pos["mark"].append("LASTSECTION")
            lastpos = pos
        lastpos["mark"].append("LASTSENTENCE")

    def __CalcKeywords(self):
        # 计算tf-idfs，取出排名靠前的20个词
        words_best = list()
        words_best = words_best + jieba.analyse.extract_tags(self.text, topK=20)
        # 提取第一段的关键词
        parts = self.text.lstrip().split("\n")
        firstpart = ""
        if len(parts) >= 1:
            firstpart = parts[0]
        words_best = words_best + jieba.analyse.extract_tags(firstpart, topK=5)
        # 提取title中的关键词
        words_best =  words_best + jieba.analyse.extract_tags(self.title, topK=3)
        # 将结果合并成一个句子，并进行分词
        text = ""
        for w in words_best:
            text = text + " " + w
        # 计算词性，提取名词和动词
        words = jieba.posseg.cut(text)
        keywords = list()
        for w in words:
            flag = w.flag
            word = w.word
            if flag.find('n') >= 0 or flag.find('v') >= 0:
                if len(word) > 1:
                    keywords.append(word)
        # 保留前20个关键词
        keywords = jieba.analyse.extract_tags(" ".join(keywords), topK=20)
        keywords = list(set(keywords))
        self.keywords = keywords

    def __CalcSentenceWeightByPos(self):
        # 计算句子的位置权重
        for sentence in self.sentences:
            mark = sentence["pos"]["mark"]
            weightPos = 0
            if "FIRSTSECTION" in mark:
                weightPos = weightPos + 2
            if "FIRSTSENTENCE" in mark:
                weightPos = weightPos + 2
            if "LASTSENTENCE" in mark:
                weightPos = weightPos + 1
            if "LASTSECTION" in mark:
                weightPos = weightPos + 1
            sentence["weightPos"] = weightPos

    def __CalcSentenceWeightByCueWords(self):
        # 计算句子的线索词权重
        index = ["总之", "总而言之", "综上所述", "归纳上述", "综上", "反正", "归纳", "一言以蔽之", "括而言之", "括而言之"]
        for sentence in self.sentences:
            sentence["weightCueWords"] = 0
        for i in index:
            for sentence in self.sentences:
                if sentence["text"].find(i) >= 0:
                    sentence["weightCueWords"] = 1

    def __CalcSentenceWeight(self):
        #初步计算句子权重
        self.__CalcSentenceWeightByPos()
        self.__CalcSentenceWeightByCueWords()
        for sentence in self.sentences:
            sentence["weight"] = 0.2 * sentence["weightPos"] + 0.8 * sentence["weightCueWords"]

    def __FirstSummary(self):
        # 清空变量
        self.first_summary = ''
        self.keywords = list()
        self.sentences = list()
        self.summary = list()

        # 调用方法，分别计算关键词、分句，计算权重
        self.__CalcKeywords()
        self.__SplitSentence()
        self.__CalcSentenceWeight()

        # 对句子的权重值进行排序
        self.sentences = sorted(self.sentences, key=lambda k: k['weight'], reverse=True)
        # 根据排序结果，取排名占前X%的句子作为摘要
        # print(len(self.sentences))
        for i in range(len(self.sentences)):
            if i < self.first_ratio * len(self.sentences):
                sentence = self.sentences[i]
                self.summary.append(sentence["text"])
        self.first_summary = ''.join(self.summary)

    def __KnnSmooth(self):
        #knn平滑
        sen_score = list(self.corr_score.values())
        sen_list = list(self.corr_score.keys())
        avg_score = np.sum(sen_score)/len(sen_score)
        for i in range(self.smoth_win):
            sen_score.insert(0, avg_score)
            sen_score.append(avg_score)
        for i, sen in enumerate(sen_list):
            score = self.corr_score[sen]
            if i <= len(sen_list):
                for z in range(1, self.smoth_win+1):
                    score += score + sen_score[i+self.smoth_win+z] + sen_score[i+self.smoth_win-z]
                self.corr_score[sen] = score/(2*self.smoth_win+1)

    def __GetCorr(self, text):
        if isinstance(text, list): text = ' '.join(text)
        stop_words = load_stop_words(cf.start_stop_words)
        sub_sentences = split_text(text)
        word_frequency = read_word_frequency(cf.word_frequency)
        model = read_word2vec(cf.words_model)
        text = cut(text, stop_words).split()
        sen_vec = SIF_sentence_embedding(text, word_frequency, model)
        #计算标题词向量
        if self.title != '':
            sub_title = cut(self.title, stop_words).split()
            sen_title = SIF_sentence_embedding(sub_title, word_frequency, model)
        #计算句子向量和词向量的余弦相似度，作为句子的评分
        for sen in sub_sentences:
            words = cut(sen, stop_words).split()
            if words != []:
                sub_sen_vec = SIF_sentence_embedding(words, word_frequency, model)
                if sub_sen_vec != '':
                    get_text_score = cosine(sen_vec, sub_sen_vec)
                    if self.title == '':
                        get_title_score = 0
                    else:
                        get_title_score = cosine(sen_title, sub_sen_vec)
                    self.corr_score[sen] = self.text_ratio*get_text_score + self.title_ratio*get_title_score
        self.__KnnSmooth()
        self.score = sorted(self.corr_score.items(), key=lambda x: x[1], reverse=True)

    def GetSummarization(self):
        #第二次得到摘要
        self.__FirstSummary()
        text = self.first_summary
        self.__GetCorr(text)
        text = self.first_summary
        sub_sentences = split_text(text)
        ranking_sentences = self.score
        selected_sen = set()
        for i in range(self.mun_len):
            selected_sen.add(ranking_sentences[i][0])
        #current_sen = ''
        # for sen, _ in ranking_sentences:
        #     if len(current_sen) < sum_len:
        #         selected_sen.add(sen)
        #     else:
        #         break

        summarized = []
        for sen in sub_sentences:
            if sen in selected_sen:
                summarized.append(sen)
        if len(summarized[-1]) < 5:
            summarized.pop(-1)
        if summarized[-1][-1] != ['。', '?', '!']:
            ch = '。'
            summarized[-1] = summarized[-1][0: -1] + ch
        end_summariry = ''.join(summarized)
        return end_summariry

if __name__ == '__main__':
    text = '虽然至今夏普智能手机在市场上无法排得上号，已经完全没落，并于 2013 年退出中国市场，但是今年 3 月份官方突然宣布回归中国，预示着很快就有夏普新机在中国登场了。那么，第一款夏普手机什么时候登陆中国呢？又会是怎么样的手机呢？\r\n近日，一款型号为 FS8016 的夏普神秘新机悄然出现在 GeekBench 的跑分库上。从其中相关信息了解到，这款机子并非旗舰定位，所搭载的是高通骁龙 660 处理器，配备有 4GB 的内存。骁龙 660 是高通今年最受瞩目的芯片之一，采用 14 纳米工艺，八个 Kryo 260 核心设计，集成 Adreno 512 GPU 和 X12 LTE 调制解调器。\r\n当前市面上只有一款机子采用了骁龙 660 处理器，那就是已经上市销售的 OPPO R11。骁龙 660 尽管并非旗舰芯片，但在多核新能上比去年骁龙 820 强，单核改进也很明显，所以放在今年仍可以让很多手机变成高端机。不过，由于 OPPO 与高通签署了排他性协议，可以独占两三个月时间。\r\n考虑到夏普既然开始测试新机了，说明只要等独占时期一过，夏普就能发布骁龙 660 新品了。按照之前被曝光的渲染图了解，夏普的新机核心竞争优势还是全面屏，因为从 2013 年推出全球首款全面屏手机 EDGEST 302SH 至今，夏普手机推出了多达 28 款的全面屏手机。\r\n在 5 月份的媒体沟通会上，惠普罗忠生表示：“我敢打赌，12 个月之后，在座的各位手机都会换掉。因为全面屏时代的到来，我们怀揣的手机都将成为传统手机。”'
    title = "如家道歉遇袭事件称努力改正 当事人曾就职浙江某媒体"
    getSumary = GetSummary()
    getSumary.SetText(title, text)
    summary = getSumary.GetSummarization()
    print(summary)