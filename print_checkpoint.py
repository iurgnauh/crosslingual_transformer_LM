# !/usr/bin/python
# -- coding:utf-8 --

import sys
from tensorflow.python import pywrap_tensorflow

assert len(sys.argv) == 2, 'python xx.py checkpoint_path'
checkpoint_path = sys.argv[1]

reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

size_all = 0.0
for k in var_to_shape_map:
    shape = var_to_shape_map[k]
    size = 4
    for i in shape:
        size = size * i
    size = size /1024/1024
    size_all += size
    print(k, shape, size)

print("Size all: %.4fM"%size_all)
