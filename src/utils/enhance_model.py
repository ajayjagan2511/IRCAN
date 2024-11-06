import torch
import json
import random
import os
from collections import Counter


def pos_list2str(pos_list):
    return '@'.join([str(pos) for pos in pos_list])


def pos_str2list(pos_str):
    return [int(pos) for pos in pos_str.split('@')]


def enhance_neurons(args, model):
    ### identify neurons
    cn_file = os.path.join(args.cn_dir, f'{args.dataset_name}-{args.model_name}-cn_bag-context.json')
    print(f"Change context neurons in: {cn_file}")
    with open(cn_file, 'r') as fr:
        cn_bag_list = json.load(fr)
    cns = []
    cn_counter = Counter()
    for cn_bag in cn_bag_list:  
        for cn in cn_bag:
            cn_counter.update([pos_list2str(cn[:2])])
    most_common_cn = cn_counter.most_common(args.enhance_cn_num)
    print(most_common_cn)
    cns = [pos_str2list(cn_str[0]) for cn_str in most_common_cn]
    print(f'The number of changed neurons: {len(cns)}')

    if args.do_random_cn:
        cns = []
        for i in range(args.enhance_cn_num):
            layer = random.randint(0, model.config.num_hidden_layers-1)
            pos = random.randint(0, model.config.intermediate_size-1)
            cns.append([layer, pos])

    ### enhance the weights of identified neurons
    for layer, pos in cns:
        with torch.no_grad():
            model.model.layers[layer].mlp.down_proj.weight[:, pos] *= args.enhance_strength
    if not args.do_random_cn:
        return model, most_common_cn
    else:
        return model, cns