import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt



fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)

tasks =  ["conll2003", "wikigold", "btc", "re3d", "gum"]
masks =["1", "2", "3", "4", "5"]

harvest = np.array([[1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0]
                    ])
# name = 'ner.png'

# Show all ticks and label them with the respective list entries
ax1.set_xticks(np.arange(len(masks)), labels=masks)
ax1.set_yticks(np.arange(len(tasks)), labels=tasks)

# Rotate the tick labels and set their alignment.
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(tasks)):
    for j in range(len(masks)):
        text = ax1.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax1.set_title("NER")
ax1.imshow(harvest)



tasks =  ["icsi", "ami", "reddit", "stack", "nyt", "emails"]
masks =["1", "2", "3", "4", "5", "6"]

harvest = np.array([[1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    ])
# name = 'sum.png'
# Show all ticks and label them with the respective list entries
ax2.set_xticks(np.arange(len(masks)), labels=masks)
ax2.set_yticks(np.arange(len(tasks)), labels=tasks)

# Rotate the tick labels and set their alignment.
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(tasks)):
    for j in range(len(masks)):
        text = ax2.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax2.set_title("SUM")
ax2.imshow(harvest)



tasks =  ["Yahoo", "AGnews", "Amazon", "Dbpedia", "Yelp"]
masks =["1", "2", "3", "4", "5"]

harvest = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    ])
# name = 'ccd.png'

# name = 'sum.png'
# Show all ticks and label them with the respective list entries
ax3.set_xticks(np.arange(len(masks)), labels=masks)
ax3.set_yticks(np.arange(len(tasks)), labels=tasks)

# Rotate the tick labels and set their alignment.
plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(tasks)):
    for j in range(len(masks)):
        text = ax3.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax3.set_title("CCD")
ax3.imshow(harvest)



tasks =  ["taxi", "hotel", "attraction", "train", "restaurant"]
masks =["1", "2", "3", "4", "5"]

harvest = np.array([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    ])

# name = 'dialogue.png'

# Show all ticks and label them with the respective list entries
ax4.set_xticks(np.arange(len(masks)), labels=masks)
ax4.set_yticks(np.arange(len(tasks)), labels=tasks)

# Rotate the tick labels and set their alignment.
plt.setp(ax4.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(tasks)):
    for j in range(len(masks)):
        text = ax4.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax4.set_title("DRG")
ax4.imshow(harvest)



tasks = ['CanonG3',
'Computer',
'DiaperChamp',
'Router',
'CreativeLabs',
'Nokia6610',
'Norton',
'restaurant',
'Nikon4300',
'HitachiRouter',
'MicroMP3',
'LinksysRouter',
'ApexAD2600e',
'ipod',
'Nokia6600',
'CanonSD500',
'Speaker',
'CanonS100',
'laptop']

masks =[str(i) for i in range(19)]


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
# name = 'asc.png'

# Show all ticks and label them with the respective list entries
ax5.set_xticks(np.arange(len(masks)), labels=masks)
ax5.set_yticks(np.arange(len(tasks)), labels=tasks)

# Rotate the tick labels and set their alignment.
plt.setp(ax5.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(tasks)):
    for j in range(len(masks)):
        text = ax5.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax5.set_title("ASC")
ax5.imshow(harvest)

# fig, ax = plt.subplots()
# im = ax.imshow(harvest)

fig.suptitle('Mask Usage')
# fig.tight_layout()
plt.savefig('mask')