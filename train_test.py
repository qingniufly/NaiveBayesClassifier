__anthor__ = 'YJ'
# -*- coding:utf-8 -*-
import os
import jieba
import nltk
import pickle

#由语料库生成数据
folder_path =  ".\\SogouC.reduced\\reduced"
folder_list = os.listdir(folder_path)
class_list = []
nClass = 0
N = 200
train_set = []
test_set = []
all_words = {}
for i in range(len(folder_list)):  #各种类别
    new_folder_path =  folder_path + os.sep + folder_list[i]
    files = os.listdir(new_folder_path)
    class_list.append(nClass)
    nClass +=1
    j = 0
    nFile = min([len(files), N])
    for file in files:  #各个文件中
        if j>N:
            break

        fobj  = open (new_folder_path+os.sep+file, 'r')
        try:
            raw = fobj.read()
        except UnicodeDecodeError:
            pass
        word_cut = jieba.cut(raw, cut_all=False)    #此处有坑:variable referenced before assigment
        word_list = list(word_cut)
        for word in word_list:
            if word in all_words:
                all_words[word] += 1
            else:
                all_words[word] = 1
        if j > 0.2*nFile: #20%测试，80%训练集
            train_set.append((word_list, class_list[i]))
        else:
            test_set.append((word_list, class_list[i]))
        fobj.close()
        j += 1
        print('Folder', i, '-file-', j, 'all_words length=', len(all_words))
        
#根据word的词频排序
all_words_list = sorted(all_words.items(), key=lambda e:e[1], reverse=True)
##for i in range(100):
##    print (all_words_list[i], i)
#去除停止词
with open('.\\chinese_stopword.txt', 'r', encoding='utf-8',errors='ignore')\
     as stopwords_file:
    stopwords_list = []
    for line in stopwords_file:
        stopwords_list.append(line[:len(line)-1])
    stopwords_file.close()
all_words_list_copy = []
for word in all_words_list:
    if word[0] not in stopwords_list:
        all_words_list_copy.append(word)
all_words_list = all_words_list_copy[:]
print(len(all_words_list))



def words_dict():
    word_features = []
    for word in all_words_list:
        word_features.append(word[0])
    return word_features

def document_features(document):
    document_words = set(document) #文件中所有词的集合
    features = {} 
    for word in word_features:
        features['contains(%s)'%word] = (word in document_words)
    return features

def text_classifier():
    train_data = [(document_features(d),c) for (d,c) in train_set]
    print('train number', len(train_set), '\ntest number', len(test_set))
    #nltk朴素贝叶斯分类器
    print('Training...')
    classifier = nltk.NaiveBayesClassifier.train(train_data)
    del train_data
    print('Testing...')
    test_data = [(document_features(d), c) for (d,c) in test_set]
    test_error = nltk.classify.accuracy(classifier, test_data)

    #缓存classifier在cache
    with open('.cache_file', 'wb') as cache_file:
        pickle.dump(classifier, cache_file)
        pickle.dump(word_features, cache_file)
        cache_file.close()

    return test_error

#正片
if __name__ == '__main__':
    word_features = words_dict()
    accuracy = text_classifier()
    print ('test accuracy：', accuracy)
    wait = input('Press Enter to continue')
