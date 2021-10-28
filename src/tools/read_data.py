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
        reviews = review_file.readlines()
        for review in reviews:
            current_aspect = []
            current_sentiment = []
            if review[:2] != '##' and '##' in review and '[' in review and \
                    ('+' in review.split('##')[0] or '-' in review.split('##')[0]):
                print(review.split('##')[0])
                sentences.append(review.split('##')[1][:-1].replace('\t',' '))

                #aspects: may be more than one
                aspect_str = review.split('##')[0]
                if ',' in aspect_str:
                    aspect_all = aspect_str.split(',')
                    for each_aspect in aspect_all:
                        # print('each_aspect.split([)[0]: ',each_aspect.split('[')[0])
                        current_aspect.append(each_aspect.split('[')[0])
                        current_sentiment.append(each_aspect.split('[')[1][0])

                elif ',' not in aspect_str:
                    # print('aspect_str.split([)[0]: ',aspect_str.split('[')[0])
                    current_aspect.append(aspect_str.split('[')[0])
                    current_sentiment.append(aspect_str.split('[')[1][0])
                num_sentence+=1
                aspects.append(current_aspect)
                sentiments.append(current_sentiment)



    # print('sentences: ',sentences)
    print('sentiments: ',sentiments)
    print('aspects: ',aspects)

    print('num_sentence: ',num_sentence)

    return sentences,aspects,sentiments


def read_semseval_2016(location):
    num_sentence = 0
    sentences = []
    sentiments = []
    aspects  = []


    root = ET.parse(location).getroot()

    for sentence in root.findall('./Review/sentences/sentence'):
        if 'Opinions' not in [_.tag for _ in sentence.iter()]:
            continue
        num_sentence+=1
        for child in sentence.iter(): #iter includes itself
            if child.tag == 'text':
                sentences.append(child.text)
            elif child.tag == 'Opinions':
                sentiment = []
                aspect = []
                for opinion in child.iter():
                    if opinion.tag == 'Opinion':
                        sentiment.append(opinion.get('polarity'))
                        aspect.append(opinion.get('target'))
                sentiments.append(sentiment)
                aspects.append(aspect)

    # print(sentences[:10])
    print(sentiments[:10])
    print(aspects[:10])

    print('num_sentence: ',num_sentence)

    return sentences,aspects,sentiments



def read_xu_semseval16(location):
    sentences = []
    sentiments = []
    aspects  = []

    with open(location,'r') as review_file:
        instance = json.load(review_file)
        for id, ins in instance['data'].items():
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


def write_to_file(file_name,sentences,aspects,sentiments):

    if 'XuSemEval' in file_name:
        with open(file_name,'w') as file_write, \
            open(file_name+'_sentence','w') as file_sentence, \
            open(file_name+'_aspect','w') as file_aspecte, \
            open(file_name+'_sentiment','w') as file_sentiment:
            for i in range(len(sentences)):
                file_write.writelines(str(sentences[i]) + '\t' + str(aspects[i]) + '\t' + str(sentiments[i]) + '\n')
                file_sentence.writelines(str(sentences[i]) + '\n')
                file_aspecte.writelines(str(aspects[i]) + '\n')
                file_sentiment.writelines(str(sentiments[i]) + '\n')

    elif 'Bing' in file_name:
        sentence_nums = len(sentences)

        train_nums = int(sentence_nums *0.8)
        valid_nums = int(sentence_nums *0.1)
        test_nums = int(sentence_nums *0.1)

        print('train_nums: ',train_nums)
        print('valid_nums: ',valid_nums)
        print('test_nums: ',test_nums)

        print('Train')
        with open(file_name+'_train','w') as file_write, \
            open(file_name+'_train_sentence','w') as file_sentence, \
            open(file_name+'_train_aspect','w') as file_aspecte, \
            open(file_name+'_train_sentiment','w') as file_sentiment:
            for i in range(len(sentences)):
                for i_i in range(len(aspects[i])):
                    file_write.writelines(str(sentences[i]) + '\t' + str(aspects[i][i_i]) + '\t' + str(sentiments[i][i_i]) + '\n')
                    file_sentence.writelines(str(sentences[i]) + '\n')
                    file_aspecte.writelines(str(aspects[i][i_i]) + '\n')
                    file_sentiment.writelines(str(sentiments[i][i_i]) + '\n')



                if i == train_nums:
                    break

        print('Valid')
        with open(file_name+'_valid','w') as file_write, \
            open(file_name+'_valid_sentence','w') as file_sentence, \
            open(file_name+'_valid_aspect','w') as file_aspecte, \
            open(file_name+'_valid_sentiment','w') as file_sentiment:
            for i in range(len(sentences)):
                x = i+train_nums
                for i_i in range(len(aspects[x])):
                    file_write.writelines(str(sentences[x]) + '\t' + str(aspects[x][i_i]) + '\t' + str(sentiments[x][i_i]) + '\n')
                    file_sentence.writelines(str(sentences[x]) + '\n')
                    file_aspecte.writelines(str(aspects[x][i_i]) + '\n')
                    file_sentiment.writelines(str(sentiments[x][i_i]) + '\n')

                if i == valid_nums:
                    break

        print('Test')
        with open(file_name+'_test','w') as file_write, \
            open(file_name+'_test_sentence','w') as file_sentence, \
            open(file_name+'_test_aspect','w') as file_aspecte, \
            open(file_name+'_test_sentiment','w') as file_sentiment:
            for i in range(len(sentences)):
                x = i+train_nums+valid_nums

                for i_i in range(len(aspects[x])):
                    file_write.writelines(str(sentences[x]) + '\t' + str(aspects[x][i_i]) + '\t' + str(sentiments[x][i_i]) + '\n')
                    file_sentence.writelines(str(sentences[x]) + '\n')
                    file_aspecte.writelines(str(aspects[x][i_i]) + '\n')
                    file_sentiment.writelines(str(sentiments[x][i_i]) + '\n')

                if x == len(sentences)-1:
                    break


    elif 'SemEval16' in file_name:
        with open(file_name,'w') as file_write, \
            open(file_name+'_sentence','w') as file_sentence, \
            open(file_name+'_aspect','w') as file_aspecte, \
            open(file_name+'_sentiment','w') as file_sentiment:
            for i in range(len(sentences)):
                for i_i in range(len(aspects[i])):
                    file_write.writelines(str(sentences[i]) + '\t' + str(aspects[i][i_i]) + '\t' + str(sentiments[i][i_i]) + '\n')
                    file_sentence.writelines(str(sentences[i]) + '\n')
                    file_aspecte.writelines(str(aspects[i][i_i]) + '\n')
                    file_sentiment.writelines(str(sentiments[i][i_i]) + '\n')

if __name__ == "__main__":


    domains = ['Speaker','Router','Computer']
    for domain in domains:
        print('Read Bing3domains ' + domain)

        sentences,aspects,sentiments = read_bing_reviews('./data/Review3Domains/'+domain+'.txt')
        write_to_file('./data/AscAll/Bing3domains_'+domain,sentences,aspects,sentiments)

    domains = ['Nokia6610','NikonCoolpix4300','CreativeLabsNomadJukeboxZenXtra40GB','CanonG3','ApexAD2600Progressive']

    for domain in domains:
        print('Read Bing5domains ' + domain)

        sentences,aspects,sentiments = read_bing_reviews('./data/Review5Domains/'+domain+'.txt')
        write_to_file('./data/AscAll/Bing5domains_'+domain,sentences,aspects,sentiments)

    print('Read Bing9domains')
    domains = ['CanonPowerShotSD500','CanonS100','DiaperChamp','HitachiRouter','ipod', \
               'LinksysRouter','MicroMP3','Nokia6600','Norton']

    for domain in domains:
        print('Read Bing9domains ' + domain)
        sentences,aspects,sentiments = read_bing_reviews('./data/Review9Domains/'+domain+'.txt')
        write_to_file('./data/AscAll/Bing9domains_'+domain,sentences,aspects,sentiments)

    # read from Xu
    years = ['14','16']
    for year in years:
        if year == '16':
            domains = ['rest',]
        elif year == '14':
            domains = ['rest','laptop']

        for domain in domains:
            print('Read SemEval ' + year + ' ' + domain)
            for type in ['train','valid','test']:
                if year == '16':
                    sentences,aspects,sentiments = read_xu_semseval16('./data/XuSemEval/'+year+'/'+domain+'/'+type+'.json')
                elif year == '14':
                    sentences,aspects,sentiments = read_xu_semseval14('./data/XuSemEval/'+year+'/'+domain+'/'+type+'.json')

                write_to_file('./data/AscAll/XuSemEval'+year+'_'+domain+'_'+type,sentences,aspects,sentiments)



    # read from original data
    # domains = ['Restaurants_train','Restaurants_test','Restaurants_valid']
    #
    # for domain in domains:
    #     print('Read SemEval 2016 ' + domain)
    #     sentences,aspects,sentiments = read_semseval_2016('./data/SemEval2016/'+domain+'.xml')
    #     write_to_file('./data/AscAll/SemEval16-'+domain,sentences,aspects,sentiments)
    #
    # domains = ['Laptops_train','Restaurants_train','Laptops_test','Restaurants_test','Laptops_valid','Restaurants_valid']
    # for domain in domains:
    #     print('Read SemEval 2014 ' + domain)
    #     sentences,aspects,sentiments = read_semseval_2016('./data/SemEval2014/'+domain+'.xml')
    #     write_to_file('./data/AscAll/SemEval14-'+domain,sentences,aspects,sentiments)
