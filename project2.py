import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk
import glob
import io
from sklearn.feature_extraction.text import CountVectorizer

import re
import os
import pdb
import sys
list_1=[]
new=[]
redact_list=[]
data=[]
with_s=[]
names=[]
length_withoutspaces=[]
spaces=[]
words_file=[]
words_word=[]
new_names=[]
with_symbol=[]
without_symbol=[]
list_2=[]
lenofeachword=[]
lenofredfirst=[]
lenofredsec=[]
lenofredthird=[]
datared=[]
names1=[]
def extraction():
    for thefile in glob.glob('train/*.txt'):
        with io.open(thefile,'r',encoding='utf-8') as f:
            text=f.read()
            data.append(text)
    #for counter in data:
        #print(nltk.word_tokenize(counter))
    return data
def extractredact():
    for thefile1 in glob.glob('test/*.txt'):
        with io.open(thefile1,'r',encoding='utf-8') as f1:
            textr=f1.read()
            datared.append(textr)
    return datared


def extract_names(d):
    for each in d:
         for sent in sent_tokenize(each):
            for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
                if hasattr(chunk,'label') and chunk.label()=='PERSON':
                     names.append(' '.join(c[0] for c in chunk.leaves()))
         words_file.append(len(nltk.word_tokenize(each)))
    return names
firstwordlength=[]
secondwordlength=[]
thirdwordlength=[]
def extract_features(e):
    for each_1 in e:
        length_withoutspaces.append(len(each_1)-each_1.count(' '))
        spaces.append(each_1.count(' '))
        words_word.append(len(nltk.word_tokenize(each_1)))
        wordslength=[] #storing each word length
        for word in nltk.word_tokenize(each_1):
            wordslength.append(len(word))
        if(len(wordslength)<3):
            wordslength.append(0)
            wordslength.append(0)
        firstwordlength.append(wordslength[0])
        secondwordlength.append(wordslength[1])
        thirdwordlength.append(wordslength[2])

    #print(lenofeachword)

    #print(length_withoutspaces)
    return length_withoutspaces,spaces,words_word,firstwordlength,secondwordlength,thirdwordlength
redactedlength_withoutspaces=[]
redactedspaces=[]
redactedwords_word=[]
firstredactedwordlength=[]
secondredactedwordlength=[]
thirdredactedwordlength=[]

testingnames=[]
def test(g):
    without_symbol=0
    with_symbol=0
    count_l=0
    count=0
    #names1=[]
    for each in g:
         for sent in sent_tokenize(each):
            for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
                if hasattr(chunk,'label') and chunk.label()=='PERSON':
                     names1.append(' '.join(c[0] for c in chunk.leaves()))
    #testingnames.append(names1)
    #print('names are ',names1)
    #testingnames=names1.copy()
    without_symbol=names1
    names2=names1.copy()
    #print('without_symbol',without_symbol)
    for i in range(len(names2)):
        if " " in names2[i]:
            names2[i]=names2[i].replace(' ','@')
    with_symbol=names2
    #print('2 nd withour symbols',without_symbol)
    #print('with symbol',with_symbol)
    #print('number of docs',len(g))
    for each in range(len(g)):
        #print(g[each])
        for i in range(len(without_symbol)):
            #print(without_symbol[i])
            #print(with_symbol[i])
            g[each]=g[each].replace(without_symbol[i], with_symbol[i])
        #print(g[each])
        for element in with_symbol:
            element=element.split('@')
            list_2.append(element)
            if len(element)==1:
                g[each]=g[each].replace(element[0],'█'*len(element[0]))
            else:
                for item in element:
                    g[each]=g[each].replace(item,'█'*len(item))
        all=g[each].split(" ")
        for item in all:
            wordcountlength=[]
            if item.__contains__('█') and item.__contains__('@'):
                #print('more than one word',item)
                separatorcount=item.count('@')
                redactedspaces.append(separatorcount)
                lengthofnamewithoutspace=len(item)-separatorcount
                redactedlength_withoutspaces.append(lengthofnamewithoutspace)
                #print('more than one word',item)
                key=item.split('@')
                for li in range(len(key)):
                    #print(key[li].count('█'))
                    wordcountlength.append(key[li].count('█'))
                lengthofname=sum(wordcountlength)
                #print('name length',lengthofname)
                #print(separatorcount)
                #print("number of words in a redacted word:",separatorcount+1)
                redactedwords_word.append(separatorcount+1)
            elif (item.__contains__('█')):
                #print('one word',item)
                # count_l=count_l+1
                #print('one word',item)
                #print(len(item))
                redactedlength_withoutspaces.append(len(item))
                redactedspaces.append(0)
                redactedwords_word.append(1)
                wordcountlength.append(len(item))
            #print('word count length is ',wordcountlength)
            if(len(wordcountlength)>=1 and len(wordcountlength)<5):
                wordcountlength.append(0);wordcountlength.append(0)
                firstredactedwordlength.append(wordcountlength[0])
                secondredactedwordlength.append(wordcountlength[1])
                thirdredactedwordlength.append(wordcountlength[2])
    return redactedlength_withoutspaces,redactedspaces,redactedwords_word,firstredactedwordlength,secondredactedwordlength,thirdredactedwordlength

import numpy as np
data_1=extraction()

y_train=extract_names(data_1)
print('extracted names',y_train)
y_train=np.array(y_train)
print('after numpy',y_train)
x=list(extract_features(names))
print(x)
y=list(zip(*x))
for i in range(len(y)):
    y[i]=list(y[i])
x_train=np.array(y)
cv=CountVectorizer()
print(x_train)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
#from sklearn.neighbors import KNeighborsClassifier
#model1=KNeighborsClassifier(n_neighbors=5).fit(x_train,y_train)
#Testing
datared=extractredact()
print('testing data',datared)
x_test=list(test(datared))
print(x_test)
print(x)
x_test=list(zip(*x_test))
for i in range(len(x_test)):
    x_test[i]=list(x_test[i])
x_test=list(x_test)
x_test=np.array(x_test)
print(x_test)
print(len(x_test))
y_pred=model.predict(x_test)
#y1_pred=model1.predict(x_test)
#print('knn',y1_pred)
print(list(y_pred))
#print(len(names1))
new=y_pred[:len(names1)]
'''for some_rand in range(len(datared)):
    for k in new:
        if datared[some_rand].find(k):
            print('document',some_rand)
            print(k)'''
from sklearn.metrics import accuracy_score
print('The accuracy score is:',accuracy_score(names1,new))

f=open('output.txt')
for item in y_pred:
    f.write(item)













