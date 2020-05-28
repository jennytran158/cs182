import json, sys
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from tensorflow import keras
import os
import re
import time
import numpy as np
from pathlib import Path
DATA_COLUMN = 'text'
LABEL_COLUMN = 'stars'
label_list = [1,2,3,4,5]


def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
    bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
    bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
      "output_weights", [num_labels-1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [num_labels-1], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):

        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probs = tf.nn.sigmoid(logits)
        log_probs = tf.math.log(tf.clip_by_value(probs, 1e-10, 1.0))
        minus_log_probs = tf.math.log(tf.clip_by_value(1-probs, 1e-10, 1.0))


        embed = tf.constant([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]],tf.float32)
        embed_labels = tf.nn.embedding_lookup(embed,labels)

#         constraints_loss = constraints_loss1 + constraints_loss2 + constraints_loss3 + constraints_loss4
        cond = tf.cast(probs < 0.5, tf.int32)
        cond = tf.concat([cond,tf.ones([tf.shape(cond)[0],1], tf.int32)],axis = -1)
        predicted_labels = tf.argmax(cond, axis=-1, output_type=tf.int32)
#         predicted_labels = tf.where(tf.equal(predicted_labels, 0), tf.ones_like(predicted_labels), predicted_labels)
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = tf.reduce_sum(embed_labels*log_probs + (1-embed_labels)*minus_log_probs, axis=-1)
        loss = -tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, probs,predicted_labels)
# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:
            (loss, predicted_labels, probs,log) = create_model(
                  is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.compat.v1.metrics.accuracy(label_ids, predicted_labels)
                return {
                      "accuracy": accuracy,
                }
            eval_metrics = metric_fn(label_ids, predicted_labels)
            accuracy = tf.compat.v1.metrics.accuracy(label_ids, predicted_labels)
            tf.summary.scalar('accuracy', accuracy[1])
            if mode == tf.estimator.ModeKeys.TRAIN:
                probs
                training_hooks=[
                    tf.estimator.LoggingTensorHook(
                    tensors={'accuracy':accuracy[1],"probs":probs,"log":log,"labels":label_ids
                            }, every_n_iter=SAVE_SUMMARY_STEPS)
                ]
                return tf.estimator.EstimatorSpec(mode=mode,
                  loss=loss,
                  train_op=train_op,
                  eval_metric_ops=eval_metrics,
                  training_hooks=training_hooks)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                  loss=loss,
                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, p) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
            predictions = {
              'probabilities': p,
              'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn
    # Return the actual model function in the closure

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1
SAVE_SUMMARY_STEPS = 1
# Compute # train and warmup steps from batch size
num_train_steps = 1
num_warmup_steps = 1
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()
if not(os.path.exists('bert')):
        os.makedirs('bert')


MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1
SAVE_SUMMARY_STEPS = 1
# Compute # train and warmup steps from batch size
num_train_steps = 1
num_warmup_steps = 1
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()
if not(os.path.exists('bert')):
        os.makedirs('bert')




def eval(text,model_dir):
    model_dir = os.path.join('bert', model_dir1)
    run_config1 = tf.estimator.RunConfig(
        model_dir=model_dir1,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
    model_fn1 = model_fn_builder(
      num_labels=len(label_list),
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps)
    estimator1 = tf.estimator.Estimator(
      model_fn=model_fn1,
      config=run_config1,
      params={"batch_size": BATCH_SIZE})
	# This is where you call your model to get the number of stars output
    val_InputExamples1 = text.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                   text_a = x[DATA_COLUMN],
                                                                   text_b = None,
                                                                   label = 1), axis = 1)

    val_features1 = bert.run_classifier.convert_examples_to_features(val_InputExamples1, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn1 = run_classifier.input_fn_builder(features=val_features1, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)

    predictions11 = estimator1.predict(predict_input_fn1)
    pred_list11 = []
    for pred in predictions11:
        pred_list11.append(label_list[pred['labels']])
    pred_list11 = np.array(pred_list11)
    return pred_list11

if len(sys.argv) > 1:
    validation_file = sys.argv[1]
    model_dir = sys.argv[2]

    val = pd.read_json(validation_file,lines=True)
    predicted = eval(val,model_dir)
    df = pd.DataFrame({"review_id": val['review_id'], "predicted_stars": predicted})
    df.to_json("output.jsonl", orient="records",lines=True)
    print("Output prediction file written")

else:
	print("No files")
