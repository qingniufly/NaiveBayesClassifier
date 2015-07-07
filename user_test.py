__author__ = 'YJ'
# -*-coding:utf-8-*-
import pickle
import jieba
import nltk
import os

def document_features(document):
    document_words = set(document) #文件中所有词的集合
    features = {}
    for word in word_features:
        features['contains(%s)'%word] = (word in document_words)
    return features

user_folder = ".\\user_texts"
catagories = {0:'IT', 1:'体育', 2:'军事'}
with open('.cache_file', 'rb') as cache_file:
    classifier = pickle.load(cache_file)
    word_features =pickle.load(cache_file)
    cache_file.close()

classify_result = {}
for file in os.listdir(user_folder):
    fobj = open(user_folder+os.sep+file, 'r')
    try:
        raw = fobj.read()
    except UnicodeDecodeError:
        pass
    fobj.close()
    word_cut = jieba.cut(raw, cut_all=False)    #此处有坑:variable referenced before assigment
    word_list = list(word_cut)
    user_test_data = document_features(word_list)
    classified = classifier.classify(user_test_data)
    classify_result[file] = classified
    print (file, 'is classified to',catagories[classified] )

wait = input('Press Enter to continue')
