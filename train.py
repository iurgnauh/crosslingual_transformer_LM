#!/usr/bin/env python

# ================================================================================
# Copyright 2018 Alibaba Inc. All Rights Reserved.
#
# History:
# 2018.04.27. Be created by jiangshi.lxq. Forked and adatped from tensor2tensor. 
# For internal use only. DON'T DISTRIBUTE.
# ================================================================================

#from config_biself import *
import sys
import os
from transformer import *
import tensorflow as tf
from tensorflow.python.client import device_lib
from random import shuffle
import common_utils
from importlib import import_module

tf.logging.set_verbosity(tf.logging.INFO)

## @bao dynamic import 
assert len(sys.argv) == 2, 'python xx.py config_file'
_, config_file = sys.argv
if config_file.endswith('.py'):
    config_file = config_file.rsplit('.', 1)[0]
tf.logging.info('Using config from %s'%config_file)
in_config = import_module(config_file)
params = getattr(in_config, 'params')

class EvaluationListener(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, eval_input_fn, output_dir, min_step=10000):
    #def __init__(self, estimator, eval_input_fn, output_dir, ref_file, trg_vocab_file, min_step=10000):
        self._estimator = estimator  # # type: tf.estimator.Estimator
        self._eval_input_fn = eval_input_fn  # return data iterator
        self._output_dir = output_dir  # model is stored here
        #self._eval_summary_dir = os.path.join(self._output_dir, "eval")  # summary during evaluation is stored here.
        #self._temp_result_dir = os.path.join(self._output_dir, "output")  # temp result is stored here.
        #self._best_model_dir = os.path.join(self._output_dir, "best")  # storing best model
        #self._best_bleu_file = self._best_model_dir + '/bleu.txt'

        #if not tf.gfile.Exists(self._eval_summary_dir):
        #    tf.gfile.MakeDirs(self._eval_summary_dir)
        #if not tf.gfile.Exists(self._temp_result_dir):
        #    tf.gfile.MakeDirs(self._temp_result_dir)
        #if not tf.gfile.Exists(self._best_model_dir):
        #    tf.gfile.MakeDirs(self._best_model_dir)

        self._min_step = min_step
        #self._max_bleu = 0
        # self._eval_hooks = SaveEvaluationPredictionHook(output_dir=self._output_dir, ref_file=ref_file,
        #                                                 trg_vocab_file=trg_vocab_file)

        #if os.path.exists(self._best_bleu_file):
        #    with open(self._best_bleu_file, 'r') as f:
        #        for line in f:
        #            self._max_bleu = float(line.strip())

    def _should_trigger(self, global_steps):
        if global_steps > self._min_step:
            return True
        else:
            return False

    def before_save(self, session, global_step_value):
        pass

    def after_save(self, session, global_step_value):
        if self._should_trigger(global_step_value):
            tf.logging.info("Importing parameters for evaluation at step {0}...".format(global_step_value))
            for eval_name in self._eval_input_fn:
                tf.logging.info("============== Evaluating %s ============="%eval_name)
                eval_result = self._estimator.evaluate(self._eval_input_fn[eval_name],
                                         checkpoint_path=tf.train.latest_checkpoint(self._output_dir))
                tf.logging.info("Done.")
                tf.logging.info(eval_result)
                #for eval_result in self._estimator.predict(self._eval_input_fn[eval_name],
                #                            checkpoint_path=tf.train.latest_checkpoint(self._output_dir),
                #                            yield_single_examples=False):
                #    tf.logging.info(eval_result)
            #loss = eval_result['loss']
            # # score = self._eval_hooks.score
            #
            #if loss is not None:
            #    common_utils.write_dict_to_summary(self._eval_summary_dir, dictionary={"Loss": loss},
            #                                       current_global_step=global_step_value)
                # if score >= self._max_bleu:
                #     self._max_bleu = score
                #     with open(self._best_bleu_file, 'w') as f:
                #         f.write(str(self._max_bleu))
                #         tf.logging.info("The best BLEU is {0}".format(self._max_bleu))
                #     tf.logging.info("Saving the best model")
                #     model_save(self._output_dir, self._best_model_dir)

def shuffle_train(train_src, train_trg):
    line_pairs = []
    for src_line, trg_line in zip(open(train_src), open(train_trg)):
        line_pairs.append((src_line, trg_line))
    shuffle(line_pairs)
    fsrc = open(train_src, 'w')
    ftrg = open(train_trg, 'w')
    for src_line, trg_line in line_pairs:
        print(fsrc, src_line.strip(), file=fsrc)
        print(trg_line.strip(), file=ftrg)

def output_hidden(model, params, outputfile='hidden.vec'):
    sent_no = 0
    ctr = 0
    with open(outputfile, 'w') as file_out:
        for filename in params['pred_src']:
            for xx in model.predict(lambda: input_fn(
                                                filename,
                                                params['vocab_src'],
                                                src_vocab_size=params['src_vocab_size'],
                                                batch_size=params["infer_batch_size"],
                                                is_shuffle=False,
                                                is_train=False
                                            )):
                #print(xx)
                hidden, input_id = xx['hidden'], xx['input']
                #print(hidden, input_id)
                #print(hidden.shape, input_id.shape)
                for ii in range(input_id.shape[0]):
                    if input_id[ii] == 0:
                        break
                    if input_id[ii] == 1:
                        ctr += 1
                    #print('%d_%d'%(sent_no, ii))
                    #print(hidden[ii, :])
                    #print(hidden[ii, :].shape)
                    file_out.write('%d_%d\t%s\n'%(sent_no, ii, ' '.join(map(str, hidden[ii, :]))))
                sent_no += 1
    tf.logging.info("%d Unk words during the hidden output"%ctr)
                
def output_model(model, params, output_path=None):
    import tensorflow_hub as tf_hub
    if output_path == None:
        output_path = "output_model_" + config_file

    module_spec = tf_hub.create_module_spec(transformer_export_fn, 
                                        [(set(), {'params':params})])
    with tf.Graph().as_default():
        module = tf_hub.Module(module_spec)
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
                                    tf.train.latest_checkpoint(params["save_path"]), 
                                    module.variable_map)
        with tf.Session() as session:
          init_fn(session)
          module.export(output_path, session=session)
    #serving_input_func = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    #                                        {'features': tf.VarLenFeature(dtype=tf.int64)})
    ##serving_input_func = tf.estimator.export.build_raw_serving_input_receiver_fn(
    ##                                        {'features': tf.variable(dtype=tf.int64, shape=(none, none))})
    #exporter = tf_hub.LatestModuleExporter("transformer_lm", serving_input_func)
    #exporter.export(model, output_path, model.latest_checkpoint())

def main(_):
    gpu_devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    if len(gpu_devices) == 0:
        params['num_gpus'] = 1
    else:
        params['num_gpus'] = len(gpu_devices)
    print(params)
    gpu_config = tf.ConfigProto()
    #gpu_config.report_tensor_allocations_upon_oom = True
    #gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4

    transformer = tf.estimator.Estimator(model_fn=transformer_model_fn,
                                        config=tf.estimator.RunConfig(
                                                save_checkpoints_steps=params['save_checkpoints_steps']//params['num_gpus'], 
                                                #session_config=gpu_config,
                                                log_step_count_steps=10000//params['num_gpus'],
                                                keep_checkpoint_max=params['keep_checkpoint_max']),
                                        model_dir=params["save_path"],
                                        params=params)

    #def eval_input_fn():

    #    return input_fn(
    #        params['dev_src'],
    #        params['vocab_src'],
    #        src_vocab_size=params['src_vocab_size'],
    #        batch_size=params["infer_batch_size"],
    #        is_shuffle=False,
    #        is_train=False
    #    )

    def train_input_fn():
        # shuffle_train(params['train_src'], params['train_trg'])
        return input_fn(
            params['train_src'],
            params['vocab_src'],
            src_vocab_size=params["src_vocab_size"],
            batch_size_words=params['train_batch_size_words'],
            max_len=params['max_len'],
            num_gpus=params['num_gpus'],
            is_shuffle=False,
            is_train=True
        )

    #def small_eval_input_fn():
    #    return input_fn(
    #        params["small_dev_src"],
    #        params['vocab_src'],
    #        src_vocab_size=params['src_vocab_size'],
    #        batch_size=params["infer_batch_size"],
    #        is_shuffle=False,
    #        is_train=False)

    eval_funcs = {}
    #for dset in ['en', 'es']:
    #for dset in params["small_dev_src"]:
    #    print('Add listener eval func for %s'%dset)
    #    eval_funcs[dset] = lambda : input_fn({dset: params['small_dev_src'][dset]},
    #                                         {dset: params['vocab_src'][dset]},
    #                                         src_vocab_size={dset: params['src_vocab_size'][dset]},
    #                                         batch_size=params["infer_batch_size"],
    #                                         is_shuffle=False,
    #                                         is_train=False)

    if 'en' in params['small_dev_src']:
        eval_funcs['en'] = lambda : input_fn({'en': params['small_dev_src']['en']},
                                             {'en': params['vocab_src']['en']},
                                             src_vocab_size={'en': params['src_vocab_size']['en']},
                                             batch_size=params["infer_batch_size"],
                                             is_shuffle=False,
                                             is_train=False)
    if 'es' in params['small_dev_src']:
        eval_funcs['es'] = lambda : input_fn({'es': params['small_dev_src']['es']},
                                             {'es': params['vocab_src']['es']},
                                             src_vocab_size={'es': params['src_vocab_size']['es']},
                                             batch_size=params["infer_batch_size"],
                                             is_shuffle=False,
                                             is_train=False)
    if 'nl' in params['small_dev_src']:
        eval_funcs['nl'] = lambda : input_fn({'nl': params['small_dev_src']['nl']},
                                             {'nl': params['vocab_src']['nl']},
                                             src_vocab_size={'nl': params['src_vocab_size']['nl']},
                                             batch_size=params["infer_batch_size"],
                                             is_shuffle=False,
                                             is_train=False)
    if 'zh' in params['small_dev_src']:
        eval_funcs['zh'] = lambda : input_fn({'zh': params['small_dev_src']['zh']},
                                             {'zh': params['vocab_src']['zh']},
                                             src_vocab_size={'zh': params['src_vocab_size']['zh']},
                                             batch_size=params["infer_batch_size"],
                                             is_shuffle=False,
                                             is_train=False)
    if 'de' in params['small_dev_src']:
        eval_funcs['de'] = lambda : input_fn({'de': params['small_dev_src']['de']},
                                             {'de': params['vocab_src']['de']},
                                             src_vocab_size={'de': params['src_vocab_size']['de']},
                                             batch_size=params["infer_batch_size"],
                                             is_shuffle=False,
                                             is_train=False)

    if 'da' in params['small_dev_src']:
        eval_funcs['da'] = lambda : input_fn({'da': params['small_dev_src']['da']},
                                             {'da': params['vocab_src']['da']},
                                             src_vocab_size={'da': params['src_vocab_size']['da']},
                                             batch_size=params["infer_batch_size"],
                                             is_shuffle=False,
                                             is_train=False)

    if 'el' in params['small_dev_src']:
        eval_funcs['el'] = lambda : input_fn({'el': params['small_dev_src']['el']},
                                             {'el': params['vocab_src']['el']},
                                             src_vocab_size={'el': params['src_vocab_size']['el']},
                                             batch_size=params["infer_batch_size"],
                                             is_shuffle=False,
                                             is_train=False)

    if 'pt' in params['small_dev_src']:
        eval_funcs['pt'] = lambda : input_fn({'pt': params['small_dev_src']['pt']},
                                             {'pt': params['vocab_src']['pt']},
                                             src_vocab_size={'pt': params['src_vocab_size']['pt']},
                                             batch_size=params["infer_batch_size"],
                                             is_shuffle=False,
                                             is_train=False)

    if 'it' in params['small_dev_src']:
        eval_funcs['it'] = lambda : input_fn({'it': params['small_dev_src']['it']},
                                             {'it': params['vocab_src']['it']},
                                             src_vocab_size={'it': params['src_vocab_size']['it']},
                                             batch_size=params["infer_batch_size"],
                                             is_shuffle=False,
                                             is_train=False)

    if 'sv' in params['small_dev_src']:
        eval_funcs['sv'] = lambda : input_fn({'sv': params['small_dev_src']['sv']},
                                             {'sv': params['vocab_src']['sv']},
                                             src_vocab_size={'sv': params['src_vocab_size']['sv']},
                                             batch_size=params["infer_batch_size"],
                                             is_shuffle=False,
                                             is_train=False)

    eval_listener = EvaluationListener(
                                        estimator=transformer,
                                        eval_input_fn=eval_funcs,
                                        output_dir=params['save_path'],
                                        min_step=params["min_start_step"]
                                    )
    #### predict next word
    #vocab_dict = {}
    #for lang in params["vocab_src"]:
    #    vocab_dict[lang] = {}
    #    with open(params["vocab_src"][lang], 'r') as file_in:
    #        for idx, word in enumerate(file_in):
    #            vocab_dict[lang][idx] = word.rstrip('\n')

    #for xx in transformer.predict(lambda : input_fn(params['small_dev_src'],
    #                                                params['vocab_src'],
    #                                                src_vocab_size=params['src_vocab_size'],
    #                                                batch_size=params["infer_batch_size"],
    #                                                is_shuffle=False,
    #                                                is_train=False), yield_single_examples=False):
    ##for xx in transformer.predict(train_input_fn, yield_single_examples=False):
    #    #print(xx)
    #    for lang in params["vocab_src"]:
    #        print('======================= %s ======================'%lang)
    #        #print(xx['input_%s'%lang])
    #        #for k in xx:
    #        #    if k.startswith(lang):
    #        #        print(xx[k])

    #        input_idxs = xx['input_%s'%lang]
    #        for ii in range(input_idxs.shape[0]):
    #            input_word = list(map(lambda x:vocab_dict[lang][x], input_idxs[ii]))
    #            print('==== input_word ====')
    #            print(' '.join(input_word))
    #            #for k in xx:
    #            #    if k.startswith(lang):
    #            #        output_word = map(lambda x:vocab_dict[k[-2:]][x], xx[k][ii])
    #            #        print('==== %s ===='%k)
    #            #        print('<> ' + ' '.join(output_word))
    #            output_words = [input_word]
    #            for k in xx:
    #                if k.startswith(lang):
    #                    output_words.append(['<>'] + list(map(lambda x:vocab_dict[k[-2:]][x], xx[k][ii]))[:-1])
    #            print(output_words)
    #            for ww in zip(*output_words):
    #                print('\t'.join(ww))
    #exit()

    if params['output_hidden'] == True:
        #output_hidden(transformer, params)
        output_model(transformer, params)
        exit()
    if params['eval_big'] == True:
        for dset in params["dev_src"]:
            print('========== Eval %s ========='%dset)
            eval_result = transformer.evaluate(
                lambda : input_fn(
                        {dset: params['dev_src'][dset]},
                        {dset: params['vocab_src'][dset]},
                        src_vocab_size={dset: params['src_vocab_size'][dset]},
                        #params['vocab_src'],
                        #src_vocab_size=params['src_vocab_size'],
                        batch_size=params["infer_batch_size"],
                        is_shuffle=False,
                        is_train=False
                        )
            )
            print(eval_result)
        exit()

    ## for debug
    #for xx in transformer.predict(small_eval_input_fn):
       #print(xx)
       #print(xx['atten'])
       #print(xx['hidden'].shape, xx['input'].shape, xx['atten'].shape)
       #for ii in range(xx['input'].shape[0]):
       #    if xx['input'][ii] == 0:
       #        break
       #    print('%d\t%d\t%s'%(ii, xx['input'][ii], ' '.join(map(str, xx['hidden'][ii, :]))))
    #print(transformer.evaluate(small_eval_input_fn))
    #exit()

    #with tf.contrib.tfprof.ProfileContext('./profile/', dump_steps=[10]) as pctx:
    #    transformer.train(train_input_fn, saving_listeners=[eval_listener])
    transformer.train(train_input_fn, saving_listeners=[eval_listener])

    #while True:
    #    transformer.train(train_input_fn, steps=1)
    #    for dset in eval_funcs:
    #        print('========== Eval %s ========='%dset)
    #        eval_result = transformer.evaluate(eval_funcs[dset])
    #        tf.logging.info(eval_result)

    #epoch = 0
    #while True:
    #    epoch += 1
    #    if epoch >= params['epoch']:
    #        break
    #    tf.logging.info("Epoch %i", epoch)
    #    #saving_listeners = [eval_listener]
    #    #evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(transformer, small_eval_input_fn, every_n_iter=50000)
    #    #transformer.train(train_input_fn, hooks=[evaluator])
    #    transformer.train(train_input_fn)
    #    if epoch % 1 == 0:
    #        for dset in params["dev_src"]:
    #            print('============================')
    #            print('========== Eval %s ========='%dset)
    #            print('============================')
    #            eval_result = transformer.evaluate(
    #                lambda : input_fn(
    #                        {dset: params['small_dev_src'][dset]},
    #                        {dset: params['vocab_src'][dset]},
    #                        src_vocab_size={dset: params['src_vocab_size'][dset]},
    #                        batch_size=params["infer_batch_size"],
    #                        is_shuffle=False,
    #                        is_train=False
    #                        )
    #            )
    #            print(eval_result)
    #        #print(transformer.evaluate(small_eval_input_fn))
    #    #train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    #    #eval_spec = tf.estimator.EvalSpec(input_fn=small_eval_input_fn, steps=None, throttle_secs=14400)
    #    #tf.estimator.train_and_evaluate(transformer, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run()

