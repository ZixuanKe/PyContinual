import os, sys
import numpy as np
import torch
from sklearn.utils import shuffle
import json
import xml.etree.ElementTree as ET


def read_bing_reviews(location):
    num_sentence = 0
    sentences = []
    sentiments = []
    aspects  = []

    with open(location,'r') as review_file:
        instance = json.load(review_file)
        for id, ins in instance.items():
            if ins['term']=='NULL': continue
            aspects.append(ins['term'])
            sentiments.append(ins['polarity'])
            sentences.append(ins['sentence'])
    return sentences,aspects,sentiments



def read_xu_semseval14(location):
    sentences = []
    sentiments = []
    aspects  = []

    with open(location,'r') as review_file:
        instance = json.load(review_file)
        for id, ins in instance.items():
            if ins['term']=='NULL': continue
            aspects.append(ins['term'])
            sentiments.append(ins['polarity'])
            sentences.append(ins['sentence'])
    return sentences,aspects,sentiments




def statistic(sentences,aspects,sentiments):
    '''
    #sentences
    #postive
    #negative
    #neutral
    #aspects

    '''
    print('#sentences: ',len(list(set(sentences))))
    print('#aspects: ',len(aspects))
    print('#postive: ',len([s for s in sentiments if s=='positive' or s == '+']))
    print('#negative: ',len([s for s in sentiments if s=='negative' or s == '-']))
    print('#neutral: ',len([s for s in sentiments if s=='neutral' or s == '=']))

    num_sentences=len(list(set(sentences)))
    num_aspects=len(aspects)
    num_positive=len([s for s in sentiments if s=='positive' or s == '+'])
    num_negative=len([s for s in sentiments if s=='negative' or s == '-'])
    num_neutral=len([s for s in sentiments if s=='neutral' or s == '='])


    return str(num_sentences)+' S./'+str(num_aspects)+' A./'+ \
           str(num_positive) + ' P./'+str(num_negative)+' N./'+str(num_neutral)+' Ne.'

if __name__ == "__main__":

    with open('statistic','w') as f_sta:
        domains = ['Speaker','Router','Computer']
        for domain in domains:
            print('Read Bing3domains ' + domain)
            sta=''
            for type in ['train','dev','test']:
                sentences,aspects,sentiments = read_xu_semseval14('./dat/Bing3Domains/asc/'+domain+'/'+type+'.json')
                sta += statistic(sentences,aspects,sentiments) + '\t'
            f_sta.writelines(sta+'\n')

        domains = ['Nokia6610','NikonCoolpix4300','CreativeLabsNomadJukeboxZenXtra40GB','CanonG3','ApexAD2600Progressive']
        for domain in domains:
            print('Read Bing5domains ' + domain)
            sta=''
            for type in ['train','dev','test']:
                sentences,aspects,sentiments = read_xu_semseval14('./dat/Bing5Domains/asc/'+domain+'/'+type+'.json')
                sta += statistic(sentences,aspects,sentiments) + '\t'
            f_sta.writelines(sta+'\n')

        domains = ['CanonPowerShotSD500','CanonS100','DiaperChamp','HitachiRouter','ipod', \
                   'LinksysRouter','MicroMP3','Nokia6600','Norton']

        for domain in domains:
            print('Read Bing9domains ' + domain)
            sta=''
            for type in ['train','dev','test']:
                sentences,aspects,sentiments = read_xu_semseval14('./dat/Bing9Domains/asc/'+domain+'/'+type+'.json')
                sta += statistic(sentences,aspects,sentiments) + '\t'
            f_sta.writelines(sta+'\n')


        # read from Xu
        years = ['14']
        for year in years:
            if year == '14':
                domains = ['rest','laptop']

            for domain in domains:
                sta=''
                for type in ['train','dev','test']:
                    print('Read XuSemEval ' + year + ' ' + domain + ' ' + type)
                    if year == '14':
                        sentences,aspects,sentiments = read_xu_semseval14('./dat/XuSemEval/'+year+'/'+domain+'/'+type+'.json')
                    sta += statistic(sentences,aspects,sentiments) + '\t'
                f_sta.writelines(sta+'\n')


