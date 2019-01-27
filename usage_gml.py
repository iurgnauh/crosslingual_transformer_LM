import os
import numpy as np
import tensorflow as tf
import gml.model_store.store as ms

model = ms.load(model_name="multilingual_lm_test", version="0.1")
embeddings = model(["hello world", "this is a test"], signature="en")

with tf.Session() as session:
    session.run(tf.tables_initializer())
    session.run(tf.global_variables_initializer())
    output = session.run(embeddings)
    print(output) # (batch_size, max_length, 2*7*512)
