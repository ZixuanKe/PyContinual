import gzip
import os
import json


dataset_path = '/sdb/zke4/data_sum/meeting_summarization/AMI_proprec'
output_path = '/sdb/zke4/data_sum/full'


data = {}
label = {}

def load():
    for data_type in ['train', 'val', 'test']:
        label[data_type] = []
        data[data_type] = []

        if data_type == "val":  # name of AMI is dev not val
            data_path = os.path.join(dataset_path, "dev")
        else:
            data_path = os.path.join(dataset_path, data_type)

        samples = []
        for gz_name in os.listdir(data_path):
            if 'gz' not in gz_name:
                continue
            sample_path = os.path.join(data_path, gz_name)
            with gzip.open(sample_path, 'rb') as file:
                for line in file:
                    samples.append(json.loads(line))

        for sample in samples:
            # get meetings & summary
            meeting = []
            for turn in sample['meeting']:
                sent = turn['role'] + ' ' + turn['speaker'] + " : "
                sent += ' '.join(turn['utt']['word'])
                meeting.append(sent)
            summary = ' '.join(sample['summary'])

            data[data_type].append(meeting)
            label[data_type].append(summary)



    # for data_type in ['train', 'val', 'test']:
    #     print('data_type: ',data_type)
    #     print('label: ',len(label[data_type]))
    #     print('data: ',len(data[data_type]))
    #     print('data: ',data[data_type][0])


    return data, label


def save(data, label):
    for data_type in ['train', 'val', 'test']:
        # write to stage 0 path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        source_path = os.path.join(output_path, f"{data_type}.source")
        target_path = os.path.join(output_path, f"{data_type}.target")

         # join all turns in dialogue dataset
        write_list_asline(source_path, [' '.join(x) for x in data[data_type]])
        write_list_asline(target_path, label[data_type])




def write_list_asline(path, data):
    with open(path,'w',encoding='utf-8') as file:
        for sample in data:
            file.write(sample.strip() + '\n')


data, label= load()
save(data, label)