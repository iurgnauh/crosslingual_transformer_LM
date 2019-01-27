# !/usr/bin/python
# -- coding:utf-8 --

import sys
import numpy
from tensorflow.python import pywrap_tensorflow

assert len(sys.argv) == 2, 'python xx.py checkpoint_path'
checkpoint_path = sys.argv[1]

reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

tensor2cal = []
for k in var_to_shape_map:
    if 'C' in k and 'Adam' not in k:
        #print('====================================')
        #print(k)
        tensor = reader.get_tensor(k)
        tensor2cal.append((k, tensor))
        #print(numpy.mean(tensor, 0))
        #print(numpy.std(tensor, 1))
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

    ## 
    #if tensor_1.shape[0] > 50000:
    #    return
    #norm_tensor_1 = tensor_1 / numpy.sqrt(numpy.sum(tensor_1 ** 2, 1))[:, None]
    #norm_tensor_2 = tensor_2 / numpy.sqrt(numpy.sum(tensor_2 ** 2, 1))[:, None]
    #sim = numpy.mean((1 - norm_tensor_1.dot(norm_tensor_2.T)) / 2)
    #print('Average Sim Dis %f'%sim)

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
