#!/usr/bin/env python
# -*- coding:utf8 -*-

# ================================================================================
# Copyright 2018 Alibaba Inc. All Rights Reserved.
#
# History:
# 2018.04.26. Be created by jiangshi.lxq. Forked and adatped from tensor2tensor.
# For internal use only. DON'T DISTRIBUTE.
# ================================================================================
import os

# home = os.environ['HOME']
# print(home)

params = {}
# WMT 16
# params["train_src"] = home+'/WMT16-DEEN/monolingual-language/ml.tok.bpe.de.40m'
# params["train_trg"] = home+'/WMT16-DEEN/monolingual-language/ml.tok.bpe.en.40m'
# params["dev_src"] = home+'/WMT16-DEEN/newstest2015.tok.bpe.32000.de'
# params["dev_trg"] = home+'/WMT16-DEEN/newstest2015.tok.bpe.32000.en'
# params["vocab_src"] = home+'/WMT16-DEEN/vocab.bpe.32000'
# params["vocab_trg"] = home+'/WMT16-DEEN/vocab.bpe.32000'
# params["dev_ref"] = home+'/WMT16-DEEN/newstest2015.tok.en'
# LDC 1.34M
#params["train_src"] = home+'/NMTDATA_LDC/Train/800/cn.8m.tok'
#params["train_trg"] = home+'/NMTDATA_LDC/Train/800/en.8m.tok'
#params["dev_src"] = home+'/NMTDATA_LDC/Test/source/MT02.cn.dev'
#params["dev_trg"] = home+'/NMTDATA_LDC/Test/reference/MT02/ref0'
#params["vocab_src"] = home+'/NMTDATA_LDC/vocab.cn.1.34m.30k'
#params["vocab_trg"] = home+'/NMTDATA_LDC/vocab.en.1.34m.30k'
#params["dev_ref"] = home+'/NMTDATA_LDC/Test/reference/MT02/ref'

prefix = 'test_data/mixdata/'

params["load_model"] = None

params["train_src"] = {'en': prefix + 'en_1B_train.seg', 'es': prefix + 'eswiki_train.seg'}
#params["train_src"] = prefix + 'mix_en_es_train.seg'
#params["train_trg"] = params["train_src"]
#params["dev_src"] = 'test_data/zhwiki/dev_cut'
params["small_dev_src"] = {'en': prefix + 'en_1B_dev.seg', 'es': prefix + 'eswiki_dev.seg'}
params["dev_src"] = {'en': prefix + 'en_1B_dev.seg', 'es': prefix + 'eswiki_dev.seg'}
#params["dev_src"] = 'test_data/zhwiki/dev_sorted'
#params["dev_src"] = 'test_data/zhwiki/dev_bao'
#params["dev_trg"] = params["dev_src"]
#params['pred_src'] = ['test_data/zhner/dev_cut']
params['pred_src'] = ['test_data/zhner/train.sent', 'test_data/zhner/dev.sent', 'test_data/zhner/test.sent']
params["vocab_src"] = {'en': prefix + '1B_vocab.20w', 'es': prefix + 'es_vocab.20w'}
params["mixvocab"] = False # True should be used with mixvocab file
#params["vocab_src"] = 'test_data/zhwiki/zhwiki.vocab'
#params["src_vocab_size"] = 30000
params["src_vocab_size"] = {'en':200000, 'es':200000}
#params["vocab_trg"] = params["vocab_src"]
params['init_emb'] = {'en': 'test_data/muse/en-es/vectors-en.txt', 
                      'es': 'test_data/muse/en-es/vectors-es.txt'}
params['is_train_emb'] = False # only work when init emb

#params["dev_ref"] = ''

##@bao: automatically detect
#params["num_gpus"] = 1
params["epoch"] = 100
params["save_checkpoints_steps"] = 100000#5000
params["keep_checkpoint_max"] = 5#20
#params["max_len"] = 100
params["max_len"] = 200#100
params["train_batch_size_words"] = 4096#2048
#params["optimizer"] = 'adadelta'  # adam or sgd
#params["optimizer"] = 'msgd'  # adam or sgd
params["optimizer"] = 'adam'  # adam or sgd
#params["learning_rate_decay"] = 'none'  # sqrt: 0->peak->sqrt; exp: peak->exp, used for finetune
params["learning_rate_decay"] = 'sqrt'  # sqrt: 0->peak->sqrt; exp: peak->exp, used for finetune
#params["learning_rate_peak"] = 1.0#0.0001
#params["learning_rate_peak"] = 0.01#0.0001
params["learning_rate_peak"] = 0.0001
params["warmup_steps"] = 12000#8000  # only for sqrt decay
params["decay_steps"] = 100  # only for exp decay, decay every n steps
params["decay_rate"] = 0.9  # only for exp decay
#params["trg_vocab_size"] = params["src_vocab_size"]
params["hidden_size"] = 512
params["filter_size"] = 2048
params["num_hidden_layers"] = 6
params["num_heads"] = 8
params['gradient_clip_value'] = 5.0
params["confidence"] = 0.9  # label smoothing confidence
params["prepost_dropout"] = 0.1
params["relu_dropout"] = 0.1
params["attention_dropout"] = 0.1
params["preproc_actions"] = 'n'  # layer normalization
params["postproc_actions"] = 'da'  # dropout; residual connection

params['learning_rate_emb_split'] = False
params['learning_rate_peak_emb'] = 0.0001
params['emb_pre_steps'] = 200000
params['emb_post_steps'] = 600000
params['learning_rate_inc_emb'] = 'linear'

## l2 align
params['n_emb_align'] = 50000
params['emb_align_alpha'] = 0.001 # alpha for mean, beta for var, gama for average sim
params['emb_align_beta'] = 0.0001
params['emb_align_gama'] = 0.0001
params['emb_align_gama_nsample'] = 8192
params['emb_align_gama_type'] = 'l2' # cos for cos sim, l2 for euclidean
params['emb_align_gama_reg'] = True # True for add self sim regularization

params['layer_align'] = True
params['layer_align_alpha'] = [0.001] * 7
params['layer_align_beta'] = [0.0001] * 7
params['layer_align_gama'] = [0.0001] * 7
params['layer_align_gama_type'] = 'l2' # cos for cos sim, l2 for euclidean
params['layer_align_gama_reg'] = True # True for add self sim regularization

params['layer_adv'] = True
params['layer_adv_pre_steps'] = 100000
params['layer_adv_post_steps'] = 600000
params['layer_adv_peak'] = 1.0
params['layer_adv_hidden'] = 0
params['layer_adv_dropout'] = 0.5
params['layer_adv_loss_type'] = 'entrogy' #flip for flip_gradient, entropy for entrogy
params['layer_adv_display_acc_instead_loss'] = False
params['layer_adv_ad_freq'] = 2
params['layer_adv_dis_stop_acc'] = 0.9

params['n_emb_align_identical'] = False
params['emb_align_identical_alpha'] = 100

params['lm_loss_alpha'] = {'en': 1.0, 'es': 1.0}

params['lang_spec_layer'] = None
#params['lang_spec_layer'] = 'ffn'
#params['lang_spec_layer'] = 'block'
params['align_lang_spec'] = False

## for sampled softmax
params['sampled_softmax'] = True
#params['sampled_softmax'] = False
params['n_sampled_batch'] = 8192

params["continuous_eval"] = True  # True for training
#params["beam_size"] = 6
#params["alpha"] = 0.6
#params["max_decoded_trg_len"] = 100
params["infer_batch_size"] = 1#32#100

#params["save_path"] = "model_lm_src_fb_100L_20w_sampled_softmax"
params["save_path"] = "model_lm_template"
#params["save_path"] = "model_lm_src_fb_10w"

if not os.path.exists(params["save_path"]):
    os.mkdir(params["save_path"])

#if not os.path.exists(params['output_path']):
#    os.mkdir(params['output_path'])

# forward, backward or both
params['forward'] = True
params['backward'] = True


params['output_hidden'] = False
#params['output_hidden'] = True

params['string_input'] = True

params['eval_big'] = False
#params['eval_big'] = True

# source or target
#params['src_trg'] = 'source'



#params['shared_weights'] = False
#if params['src_trg'] == 'target':
#    params["suffix"] = '_D'
#else:
#    params["suffix"] = ''

params["min_start_step"]=1

