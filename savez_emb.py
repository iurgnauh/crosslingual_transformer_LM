# !/usr/bin/python
# -- coding:utf-8 --

import sys
import numpy
from tensorflow.python import pywrap_tensorflow

assert len(sys.argv) == 3, 'python xx.py checkpoint_path save_file'
checkpoint_path = sys.argv[1]
save_file = sys.argv[2]

reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

tensor2cal = {}
for k in var_to_shape_map:
    if 'C' in k and 'Adam' not in k:
        #print('====================================')
        #print(k)
        tensor = reader.get_tensor(k)
        tensor2cal[k] = tensor

numpy.savez(save_file, **tensor2cal)
