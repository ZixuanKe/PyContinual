import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt




# tasks =  ["conll2003", "wikigold", "btc", "re3d", "gum"]
# masks =["1", "2", "3", "4", "5"]
#
# harvest = np.array([[1, 0, 0, 0, 0],
#                     [1, 0, 0, 0, 0],
#                     [1, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 0, 1, 0, 0]
#                     ])
# name = 'ner.png'




# tasks =  ["icsi", "ami", "reddit", "stack", "nyt", "emails"]
# masks =["1", "2", "3", "4", "5", "6"]
#
# harvest = np.array([[1, 0, 0, 0, 0, 0],
#                     [1, 0, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0, 0],
#                     [0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 1, 0, 0],
#                     [0, 0, 0, 0, 1, 0],
#                     ])
# name = 'sum.png'
# #


# tasks =  ["Yahoo", "AGnews", "Amazon", "Dbpedia", "Yelp"]
# masks =["1", "2", "3", "4", "5"]
#
# harvest = np.array([[1, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 0, 1, 0, 0],
#                     [0, 0, 0, 1, 0],
#                     [0, 0, 1, 0, 0],
#                     ])
# name = 'ccd.png'
#


# tasks =  ["taxi", "hotel", "attraction", "train", "restaurant"]
# masks =["1", "2", "3", "4", "5"]
#
# harvest = np.array([[1, 0, 0, 0, 0],
#                     [0, 1, 0, 0, 0],
#                     [0, 0, 1, 0, 0],
#                     [0, 0, 1, 0, 0],
#                     [0, 0, 0, 1, 0],
#                     ])
#
# name = 'dialogue.png'



tasks = ['CanonG3',
'Computer',
'DiaperChamp',
'Router',
'CreativeLabs',
'Nokia6610',
'Norton',
'restaurant',
'NikonCoolpix4300',
'HitachiRouter',
'MicroMP3',
'LinksysRouter',
'ApexAD2600e',
'ipod',
'Nokia6600',
'CanonPowerShotSD500',
'Speaker',
'CanonS100',
'laptop']

masks =[str(i+1) for i in range(19)]


harvest = np.array([

    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],


                    ])
name = 'asc.png'


#
#
# tasks = ['CanonG3',
# 'Computer',
# 'DiaperChamp',
# 'Router',
# 'CreativeLabs',
# 'Nokia6610',
# 'Norton',
# 'restaurant',
# 'NikonCoolpix4300',
# 'HitachiRouter',
# 'MicroMP3',
# 'LinksysRouter',
# 'ApexAD2600e',
# 'ipod',
# 'Nokia6600',
# 'CanonPowerShotSD500',
# 'Speaker',
# 'CanonS100',
# 'laptop']
#
# masks = [
#
# 'CanonG3',
# 'Computer',
# 'DiaperChamp',
# 'Router',
# 'CreativeLabs',
# 'Nokia6610',
# 'Norton',
# 'restaurant',
# 'NikonCoolpix4300',
# 'HitachiRouter',
# 'MicroMP3',
# 'LinksysRouter',
# 'ApexAD2600e',
# 'ipod',
# 'Nokia6600',
# 'CanonPowerShotSD500',
# 'Speaker',
# 'CanonS100',
# 'laptop']
#
# tasks.reverse()
#
#
# harvest = np.flip(np.array([
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#     [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0]
#
#                     ]),0)
#
# name = 'asc_cat.png'
#




fig, ax = plt.subplots()
# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(masks)), labels=masks)
ax.set_yticks(np.arange(len(tasks)), labels=tasks)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
# for i in range(len(tasks)):
#     for j in range(len(masks)):
#         text = ax.text(j, i, harvest[i, j],
#                        ha="center", va="center", color="w")

im = ax.imshow(harvest)
# fig.set_size_inches(10.5, 10.5)

fig.tight_layout()
plt.savefig(name)