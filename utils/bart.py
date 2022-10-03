
import torch
from networks.baselines import cat

########################################################################################################################

#TODO: need more
def lookfor_baseline_variable(self,args):
    # module.model.model.decoder.layers.8.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.0.fc1.weight


    
    self.label_list_dict = \
        {
            'conll2003': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
            'wnut2017': ['O', 'B-location', 'I-location', 'B-corporation', 'I-corporation', 'B-person', 'I-person',
                         'B-product', 'I-product', 'B-creative-work', 'I-creative-work',
                         'B-group', 'I-group'],
            'wikigold': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
            'ontonote': ['O', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC',
                         'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE',
                         'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT',
                         'B-EVENT', 'I-EVENT', 'B-WORK_OF_ART', 'I-WORK_OF_ART',
                         'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE',
                         'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME',
                         'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY',
                         'B-QUANTITY', 'I-QUANTITY', 'B-ORDINAL', 'I-ORDINAL',
                         'B-CARDINAL', 'I-CARDINAL'
                         ],
            'btc': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
            'ieer': ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG',
                     'B-PCT', 'I-PCT', 'B-MON', 'I-MON',
                     'B-TIM', 'I-TIM', 'B-DAT', 'I-DAT',
                     'B-DUR', 'I-DUR', 'B-CAR', 'I-CAR',
                     'B-MEA', 'I-MEA'
                     ],
            'ritter': ['O', 'B-person', 'I-person', 'B-geo-loc', 'I-geo-loc', 'B-facility', 'I-facility',
                       'B-company', 'I-company', 'B-sportsteam', 'I-sportsteam',
                       'B-musicartist', 'I-musicartist', 'B-product', 'I-product',
                       'B-tvshow', 'I-tvshow', 'B-movie', 'I-movie',
                       'B-other', 'I-other'
                       ],
            're3d': ['O', 'B-Person', 'I-Person', 'B-DocumentReference', 'I-DocumentReference', 'B-Location',
                     'I-Location',
                     'B-MilitaryPlatform', 'I-MilitaryPlatform', 'B-Money', 'I-Money',
                     'B-Nationality', 'I-Nationality', 'B-Organisation', 'I-Organisation',
                     'B-Quantity', 'I-Quantity', 'B-Temporal', 'I-Temporal',
                     'B-Weapon', 'I-Weapon'
                     ],
            'gum': ['O', 'B-person', 'I-person', 'B-place', 'I-place', 'B-organization', 'I-organization',
                    'B-quantity', 'I-quantity', 'B-time', 'I-time',
                    'B-event', 'I-event', 'B-abstract', 'I-abstract',
                    'B-substance', 'I-substance', 'B-object', 'I-object',
                    'B-animal', 'I-animal', 'B-plant', 'I-plant'
                    ]
    
        }
    if 'adapter_cat' in args.baseline:
        self.similarity = cat.Similarity()

    self.tsv_para = \
        ['module.model.model.'+coder+'.layers.' + str(
            layer_id) + '.output_adapters.adapters.adapter.adapter_down.'+adapter+'.adapters.capsule_net.tsv_capsules.route_weights'
         for layer_id in range(12) for coder in ['encoder','decoder'] for adapter in ['linear_down','linear_up']] + \
        ['module.model.model.encoder.layers.' + str(
            layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.route_weights'
         for layer_id in range(12) for coder in ['encoder','decoder'] for adapter in ['linear_down','linear_up']] + \
        ['module.model.model.encoder.layers.' + str(
            layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc1.' + str(
            c_t) + '.weight'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder','decoder'] for adapter in ['linear_down','linear_up']] + \
        ['module.model.model.encoder.layers.' + str(
            layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc1.' + str(
            c_t) + '.bias'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder','decoder'] for adapter in ['linear_down','linear_up']] + \
        ['module.model.model.encoder.layers.' + str(
            layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc2.' + str(
            c_t) + '.weight'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder','decoder'] for adapter in ['linear_down','linear_up']] + \
        ['module.model.model.encoder.layers.' + str(
            layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc2.' + str(
            c_t) + '.bias'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder','decoder'] for adapter in ['linear_down','linear_up']] + \
        ['module.model.model.encoder.layers.' + str(
            layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.' + str(
            c_t) + '.weight'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder','decoder'] for adapter in ['linear_down','linear_up']] + \
        ['module.model.model.encoder.layers.' + str(
            layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.' + str(
            c_t) + '.bias'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder','decoder'] for adapter in ['linear_down','linear_up']] + \
        ['module.model.model.encoder.layers.' + str(
            layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2.' + str(
            c_t) + '.weight'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder','decoder'] for adapter in ['linear_down','linear_up']] + \
        ['module.model.model.encoder.layers.' + str(
            layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2.' + str(
            c_t) + '.bias'
         for c_t in range(args.ntasks) for layer_id in range(12) for coder in ['encoder','decoder'] for adapter in ['linear_down','linear_up']]



    return self



def get_view_for_tsv(n, model_ori, args):

    # some weights should not affect eacher other, even if they are not covered by the fc mask

    t = args.ft_task

    for layer_id in range(12):
        if n == 'module.model.model.encoder.layers.' + str(
                layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.route_weights':
            return model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                t].data.view(1, -1, 1, 1)
        if n == 'module.model.model.decoder.layers.' + str(
                layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.route_weights':
            return model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[
                t].data.view(1, -1, 1, 1)
        
        if n == 'module.model.model.encoder.layers.' + str(
                layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.route_weights':
            return model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[
                t].data.view(1, -1, 1, 1)
        if n == 'module.model.model.decoder.layers.' + str(
                layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.route_weights':
            return model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[
                t].data.view(1, -1, 1, 1)


        for c_t in range(args.ntasks):
            if n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc1.' + str(c_t) + '.weight':
                return \
                model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc1.' + str(c_t) + '.bias':
                return \
                model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc2.' + str(c_t) + '.weight':
                return \
                model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc2.' + str(c_t) + '.bias':
                return \
                model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data

            if n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc1.' + str(c_t) + '.weight':
                return \
                model_ori.model.roberta.decoder.layer[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc1.' + str(c_t) + '.bias':
                return \
                model_ori.model.roberta.decoder.layer[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc2.' + str(c_t) + '.weight':
                return \
                model_ori.model.roberta.decoder.layer[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.semantic_capsules.fc2.' + str(c_t) + '.bias':
                return \
                model_ori.model.roberta.decoder.layer[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data

            if n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc1.' + str(c_t) + '.weight':
                return \
                model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc1.' + str(c_t) + '.bias':
                return \
                model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc2.' + str(c_t) + '.weight':
                return \
                model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc2.' + str(c_t) + '.bias':
                return \
                model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data

            if n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc1.' + str(c_t) + '.weight':
                return \
                model_ori.model.roberta.decoder.layer[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc1.' + str(c_t) + '.bias':
                return \
                model_ori.model.roberta.decoder.layer[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc2.' + str(c_t) + '.weight':
                return \
                model_ori.model.roberta.decoder.layer[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.semantic_capsules.fc2.' + str(c_t) + '.bias':
                return \
                model_ori.model.roberta.decoder.layer[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                    c_t].data



            for m_t in range(3):
                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs3.' + str(
                        c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs3.' + str(
                        c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs2.' + str(
                        c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs2.' + str(
                        c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs1.' + str(
                        c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs1.' + str(
                        c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data

                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs3.' + str(
                        c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs3.' + str(
                        c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs2.' + str(
                        c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs2.' + str(
                        c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs1.' + str(
                        c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data
                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.transfer_capsules.convs1.' + str(
                        c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.capsule_net.tsv_capsules.tsv[t][c_t].data

                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs3.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs3.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs2.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs2.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs1.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.model.encoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs1.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.model.encoder.layers[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data

                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs3.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs3.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs2.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs2.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs1.' + str(
                    c_t) + '.' + str(m_t) + '.weight':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data
                if n == 'module.model.model.decoder.layers.' + str(
                        layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.transfer_capsules.convs1.' + str(
                    c_t) + '.' + str(m_t) + '.bias':
                    return model_ori.model.roberta.decoder.layer[
                        layer_id].output_adapters.adapters.adapter.adapter_up.adapters.capsule_net.tsv_capsules.tsv[t][
                        c_t].data

    return 1  # if no condition is satified




def mask(model, accelerator,args):

    model_ori = accelerator.unwrap_model(model)

    masks = {}
    cat_masks = {}


    for layer_id in range(model_ori.config.num_hidden_layers):
        if 'adapter_hat' in args.baseline or 'adapter_cat' in args.baseline \
                or 'adapter_bcl' in args.baseline \
                or 'adapter_ctr' in args.baseline \
                or 'adapter_classic' in args.baseline:  # BCL included HAT

            fc1_key = 'module.model.model.encoder.layers.'+str(layer_id)+'.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1' #gfc1
            fc2_key = 'module.model.model.encoder.layers.'+str(layer_id)+'.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2' #gfc2

            masks[fc1_key], masks[fc2_key] = model_ori.model.model.encoder.layers[
                layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.mask()

            if 'adapter_cat' in args.baseline:
                cat_masks[fc1_key], cat_masks[fc2_key] = model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.cat_mask()
                masks[fc1_key], masks[fc2_key] = torch.min(masks[fc1_key],cat_masks[fc1_key]), torch.min(masks[fc2_key],cat_masks[fc2_key])
                # you already consider cat_mask
                # those used in cat is 0, min keep them as 0, in mask_back, in becomes 1-0=1, so the gradient is unblocked


            fc1_key = 'module.model.model.decoder.layers.'+str(layer_id)+'.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1' #gfc1
            fc2_key = 'module.model.model.decoder.layers.'+str(layer_id)+'.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2' #gfc2

            masks[fc1_key], masks[fc2_key] = model_ori.model.model.decoder.layers[
                layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.mask()

            if 'adapter_cat' in args.baseline:
                cat_masks[fc1_key], cat_masks[fc2_key] = model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_down.linear_down.adapters.cat_mask()
                masks[fc1_key], masks[fc2_key] = torch.min(masks[fc1_key],cat_masks[fc1_key]), torch.min(masks[fc2_key],cat_masks[fc2_key])


            fc1_key = 'module.model.model.encoder.layers.'+str(layer_id)+'.output_adapters.adapters.adapter.adapter_up.adapters.fc1' #gfc1
            fc2_key = 'module.model.model.encoder.layers.'+str(layer_id)+'.output_adapters.adapters.adapter.adapter_up.adapters.fc2' #gfc2

            masks[fc1_key], masks[fc2_key] = model_ori.model.model.encoder.layers[
                layer_id].output_adapters.adapters.adapter.adapter_up.adapters.mask()

            if 'adapter_cat' in args.baseline:
                cat_masks[fc1_key], cat_masks[fc2_key] = model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.cat_mask()
                masks[fc1_key], masks[fc2_key] = torch.min(masks[fc1_key],cat_masks[fc1_key]), torch.min(masks[fc2_key],cat_masks[fc2_key])


            fc1_key = 'module.model.model.decoder.layers.'+str(layer_id)+'.output_adapters.adapters.adapter.adapter_up.adapters.fc1' #gfc1
            fc2_key = 'module.model.model.decoder.layers.'+str(layer_id)+'.output_adapters.adapters.adapter.adapter_up.adapters.fc2' #gfc2

            masks[fc1_key], masks[fc2_key] = model_ori.model.model.decoder.layers[
                layer_id].output_adapters.adapters.adapter.adapter_up.adapters.mask()

            if 'adapter_cat' in args.baseline:
                cat_masks[fc1_key], cat_masks[fc2_key] = model_ori.model.model.encoder.layers[layer_id].output_adapters.adapters.adapter.adapter_up.adapters.cat_mask()
                masks[fc1_key], masks[fc2_key] = torch.min(masks[fc1_key],cat_masks[fc1_key]), torch.min(masks[fc2_key],cat_masks[fc2_key])


    return masks






def get_view_for(n, p, masks, config, args):
    for layer_id in range(config.num_hidden_layers):

        if 'adapter_hat' in args.baseline or 'adapter_cat' in args.baseline \
                or 'adapter_bcl' in args.baseline \
                or 'adapter_ctr' in args.baseline \
                or 'adapter_classic' in args.baseline:  # BCL included HAT
            if n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.weight':
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)

            elif n == 'module.model.model.encoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.weight':
                # print('not nont')
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.model.encoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.model.encoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.model.encoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)


            if n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.weight':
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)

            elif n == 'module.model.model.decoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.weight':
                # print('not nont')
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.model.decoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.model.decoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.model.decoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)

            if n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_up.adapters.fc1.weight':
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_up.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_up.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.model.encoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_up.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)

            elif n == 'module.model.model.encoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.fc1.weight':
                # print('not nont')
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.model.encoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.model.encoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.model.encoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)


            if n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_up.adapters.fc1.weight':
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_up.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_up.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.model.decoder.layers.' + str(
                    layer_id) + '.attention.output_adapters.adapters.adapter.adapter_up.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)

            elif n == 'module.model.model.decoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.fc1.weight':
                # print('not nont')
                return masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'module.model.model.decoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.fc1.bias':
                return masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'module.model.model.decoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                return torch.min(post, pre)
            elif n == 'module.model.model.decoder.layers.' + str(layer_id) + '.output_adapters.adapters.adapter.adapter_up.adapters.fc2.bias':
                return masks[n.replace('.bias', '')].data.view(-1)

    return None



