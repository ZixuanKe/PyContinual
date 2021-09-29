from datasets import load_dataset
#

# TODO: load schema SGD
# dataset = load_dataset('schema_guided_dstc8','schema',split='train')
# print('dataset: ',dataset['intents'])


# pre_intents ={}
# x = 0
# for intent in dataset['intents']:
#     for pre_intent in intent['name']:
#         x+=1
#         pre_intents[pre_intent] = x
#
# print('pre_intents: ',pre_intents)
# print('x: ',x)


# TODO: load schema multiwoz
dataset = load_dataset('multi_woz_v22','v2.2_active_only',split='train')
# print('dataset: ',dataset)




# TODO: load taskmaster1
# dataset = load_dataset('taskmaster1','woz_dialogs',split='train')
# # print('dataset: ',dataset['utterances'])
#
# annotation_name = {}
# x = 0
# for dialogue in dataset['utterances']:
#     for utterance in dialogue:
#         if utterance['speaker'] == 'USER':
#             for segment in utterance['segments']:
#                 if segment['annotations'][0]['name'] not in annotation_name:
#                     x+=1
#                     annotation_name[segment['annotations'][0]['name']]=x
# print('x: ',x)
# print('annotation_name: ',annotation_name)



# TODO: load taskmaster2
# dataset = load_dataset('taskmaster2','sports',split='train')
# # ['flights', 'food-ordering', 'hotels', 'movies', 'music', 'restaurant-search', 'sports']
#
# annotation_name = {}
# x = 0
# for dialogue in dataset['utterances']:
#     for utterance in dialogue:
#         if utterance['speaker'] == 'USER':
#             for segment in utterance['segments']:
#                 if segment['annotations'][0]['name'] not in annotation_name:
#                     x+=1
#                     annotation_name[segment['annotations'][0]['name']]=x
# print('x: ',x)
# print('annotation_name: ',annotation_name)
