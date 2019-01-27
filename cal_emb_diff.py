# !/usr/bin/python
# -- coding:utf-8 --

import sys
import numpy as np
from tensorflow.python import pywrap_tensorflow

assert len(sys.argv) == 2, 'python xx.py checkpoint_path'
checkpoint_path = sys.argv[1]

reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

embedding_en = None
embedding_es = None
for k in var_to_shape_map:
    if 'C_en' in k and 'Adam' not in k:
        #print('====================================')
        #print(k)
        embedding_en = reader.get_tensor(k)
        #tensor2cal.append((k, tensor))
        #print(numpy.mean(tensor, 0))
        #print(numpy.std(tensor, 1))
    if 'C_es' in k and 'Adam' not in k:
        embedding_es = reader.get_tensor(k)

#print(embedding_en)
#print(embedding_es)

def load_dict(filename):
    vocab_dict = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            vocab_dict[line.strip()] = idx

    return vocab_dict


vocab_en = load_dict('test_data/mixdata/1B_vocab.20w')
vocab_es = load_dict('test_data/mixdata/es_vocab.20w')

overlap_idx_en = []
overlap_idx_es = []

for w in vocab_en:
    if w in vocab_es:
        overlap_idx_en.append(vocab_en[w])
        overlap_idx_es.append(vocab_es[w])

assert len(overlap_idx_en) == len(overlap_idx_es)
print('Total overlapped words: {}'.format(len(overlap_idx_en)))

overlap_en = embedding_en[overlap_idx_en]
overlap_es = embedding_es[overlap_idx_es]

len_en = np.mean(np.sqrt(np.sum(overlap_en * overlap_en, -1)))# / len(overlap_en)
len_es = np.mean(np.sqrt(np.sum(overlap_es * overlap_es, -1)))# / len(overlap_es)
diff = overlap_en - overlap_es
len_diff = np.mean(np.sqrt(np.sum(diff * diff, -1)))# / len(diff)
print("Mean length of en: ", len_en)
print("Mean length of es: ", len_es)
print("Mean length of diff: ", len_diff)


#print(sum(embedding_en[overlap_idx_en[0]] * embedding_en[overlap_idx_en[0]]))
#print(sum(embedding_es[overlap_idx_es[0]] * embedding_es[overlap_idx_es[0]]))
#diff = embedding_en[overlap_idx_en[0]] - embedding_es[overlap_idx_es[0]]
#print(embedding_en[overlap_idx_en[0]] - embedding_es[overlap_idx_es[0]])
#print(sum(diff * diff))



"""
#print(tensor2cal)

def cal_dis(tensor_1, tensor_2):
    mean_1 = numpy.mean(tensor_1, 0)
    mean_2 = numpy.mean(tensor_2, 0)
    dis = numpy.sqrt(numpy.sum((mean_1 - mean_2) ** 2))
    print('Mean Dis %f'%dis)
    mean_length_1 = numpy.mean(numpy.sqrt(numpy.sum(tensor_1 ** 2, 1)))
    mean_length_2 = numpy.mean(numpy.sqrt(numpy.sum(tensor_2 ** 2, 1)))
    relative_dis = dis / mean_length_1 / mean_length_2
    print('Mean Relative Dis %f with mean length %f and %f'%(relative_dis, mean_length_1, mean_length_2))

    std_1 = numpy.std(tensor_1, 0)
    std_2 = numpy.std(tensor_2, 0)
    relative_dis = numpy.mean(2 * numpy.abs(mean_1 - mean_2) / (std_1 + std_2))
    mean_1, mean_2 = std_1, std_2
    print('Mean Relative Dis %f with mean std %f and %f'%(relative_dis, numpy.mean(mean_1), numpy.mean(mean_2)))

    dis = numpy.sqrt(numpy.sum((mean_1 - mean_2) ** 2))
    print('Std Dis %f'%dis)
    relative_dis = numpy.sqrt(numpy.sum((mean_1 - mean_2) ** 2 / numpy.abs(mean_1) /numpy.abs(mean_2)))
    print('Std Relative Dis %f with mean length %f and %f'%(relative_dis, 
                                                            numpy.mean(numpy.abs(mean_1)), 
                                                            numpy.mean(numpy.abs(mean_2))))

for idx, (name_1, tensor_1) in enumerate(tensor2cal):
    for name_2, tensor_2 in tensor2cal[idx+1:]:
        print('%s vs %s'%(name_1, name_2))
        print('====================================')
        print('=============  Cal All =============')
        print('====================================')
        cal_dis(tensor_1, tensor_2)

        print('============= Cal Top 5w =============')
        tensor_1 = tensor_1[:50000]
        tensor_2 = tensor_2[:50000]
        cal_dis(tensor_1, tensor_2)
"""
