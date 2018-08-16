import tensorflow as tf
import numpy as np
import util
from plot import plot_loss
from seq2seq import critic_model
from seq2seq import model_inputs_

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Number of Epochs
epochs = 5
# Batch Size
batch_size = 128
# Embedding Size
encoding_embedding_size = 64
decoding_embedding_size = 64
# RNN Size
rnn_size = 64
# Number of Layers
num_layers = 2
# Learning Rate
learning_rate = 0.0001
# Dropout Keep Probability
keep_probability = 0.8
display_step = 10

save_path = 'checkpoints/sup/actor/dev'
save_path_critic = 'checkpoints/sup/critic/dev'

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = util.load_preprocess('preprocess.p')
load_path_actor = util.load_params('params_actor_sup.p')

source_train = util.read_list('source_train.npy')
target_train = util.read_list('target_train.npy')
valid_size = batch_size * 10
train_source = source_train[valid_size:]
train_target = target_train[valid_size:]
valid_source = source_train[:valid_size]
valid_target = target_train[:valid_size]

test_acc_list = []

train_graph = tf.Graph()
critic_graph = tf.Graph()
actor_graph = tf.Graph()

train_sess = tf.Session(graph=train_graph)
critic_sess = tf.Session(graph=critic_graph)
actor_sess = tf.Session(graph=actor_graph)

# actor
with train_sess.as_default():
    with train_graph.as_default():
        # Load saved model and restore the saved variables
        loader = tf.train.import_meta_graph(load_path_actor + '.meta')
        loader.restore(train_sess, load_path_actor)

        input_data = train_graph.get_tensor_by_name('input:0')
        logits = train_graph.get_tensor_by_name('sample_logits:0')
        inference_logits = train_graph.get_tensor_by_name('predictions:0')
        train_logits = train_graph.get_tensor_by_name('logits:0')
        target_sequence_length = train_graph.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = train_graph.get_tensor_by_name('source_sequence_length:0')
        keep_prob = train_graph.get_tensor_by_name('keep_prob:0')
        weights = train_graph.get_tensor_by_name('weights:0')
        targets = train_graph.get_tensor_by_name('target:0')
        lr = train_graph.get_tensor_by_name('learning_rate:0')
        global_step = train_graph.get_tensor_by_name('global_step:0')
        max_target_sequence_length = train_graph.get_tensor_by_name('max_target_len:0')
        train_o = tf.get_collection('train_op')[0]
        cost = tf.get_collection('loss')[0]

# critic
with critic_sess.as_default():
    with critic_graph.as_default():
        input_data_, targets_, lr_, keep_prob_, target_sequence_length_, max_target_sequence_length_, source_sequence_length_, global_step_ = model_inputs_()
        rewards_ = tf.placeholder(tf.float32, shape=[None, None], name="rewards")
        input_shape = tf.shape(input_data)
        W = tf.Variable(tf.zeros([rnn_size, 1]), name="weight", dtype=tf.float32)
        train_logits_ = critic_model(tf.reverse(input_data_, [-1]),
                                                      targets_,
                                                      keep_prob_,
                                                      batch_size,
                                                      source_sequence_length_,
                                                      target_sequence_length_,
                                                      max_target_sequence_length_,
                                                      len(target_vocab_to_int),
                                                      len(target_vocab_to_int),
                                                      encoding_embedding_size,
                                                      decoding_embedding_size,
                                                      rnn_size,
                                                      num_layers,
                                                      target_vocab_to_int)

        # make a copy of hidden states
        training_logits_ = tf.identity(train_logits_.rnn_output)
        # linear layer for regression
        training_logits_ = tf.squeeze(tf.matmul(tf.reshape(training_logits_, [-1, rnn_size]), W))
        training_logits_ = tf.reshape(training_logits_, [batch_size, -1], name='logits_critic')

        masks_ = tf.to_float(tf.greater(targets_, 0))
        norm = tf.reduce_sum(masks_)
        with tf.name_scope("optimization_critic"):
            # MSE loss
            l_ = tf.reduce_sum(tf.multiply(tf.squared_difference(training_logits_, rewards_), masks_)) / norm

            # Optimizer
            optimizer_ = tf.train.AdamOptimizer(lr_)

            # Gradient Clipping is applied to mitigate the issue of exploding gradients
            gradients_ = optimizer_.compute_gradients(l_)
            capped_gradients_ = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients_ if grad is not None]
            train_op_ = optimizer_.apply_gradients(capped_gradients_, global_step=global_step_)

loss_list = []
reward_list = []
loss_list_critic = []
reward_valid_list = []
count = 0

with critic_sess.as_default():
        with critic_graph.as_default():
            critic_sess.run(tf.global_variables_initializer())

for epoch_i in range(epochs):
    for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                util.get_batches(train_source, train_target, batch_size,
                                 source_vocab_to_int['<PAD>'],
                                 target_vocab_to_int['<PAD>'])):
        if batch_i == 0:
            with train_sess.as_default():
                with train_graph.as_default():
                    rewards_all = 0
                    for batch_j, (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths) in enumerate(
                            util.get_batches(valid_source, valid_target, batch_size,
                                             source_vocab_to_int['<PAD>'],
                                             target_vocab_to_int['<PAD>'])):
                        batch_valid_logits = train_sess.run(
                            inference_logits,
                            {input_data: valid_sources_batch,
                             source_sequence_length: valid_sources_lengths,
                             target_sequence_length: valid_targets_lengths,
                             keep_prob: 1.0})

                        rewards_v, pre_logits = util.get_bleu(valid_targets_batch, batch_valid_logits)
                        rewards_v = np.sum(rewards_v) / batch_size
                        rewards_all += rewards_v
                    rewards_all /= 10
                    reward_valid_list.append((count, rewards_all))
                    print("reward:", rewards_all)

        count += 1
        with train_sess.as_default():
            with train_sess.graph.as_default():

                translate_logits = train_sess.run(logits, {input_data: source_batch,
                                                 target_sequence_length: targets_lengths,
                                                 source_sequence_length: sources_lengths,
                                                 keep_prob: 1.0})
        # calculate lengths of sample sentences
        lens = [[translate_logits.shape[1]] * batch_size]
        lens = np.squeeze(lens)
        # get the bleu
        rewards, pre_logits = util.get_bleu(target_batch, translate_logits)
        # copy the bleu for all time steps
        rewards_c = [[rewards[0]] * lens[0]]
        for i in range(batch_size - 1):
            temp = [[rewards[i + 1]] * lens[i + 1]]
            rewards_c = np.vstack((rewards_c, temp))
        # calculate mean reward
        reward_batch = np.sum(rewards)
        reward_batch = reward_batch / batch_size

        with critic_sess.as_default():
            with critic_sess.graph.as_default():
                # get the estimate reward
                baselines = critic_sess.run(training_logits_, { input_data_: source_batch,
                                                                targets_: translate_logits,
                                                                target_sequence_length_: lens,
                                                                source_sequence_length_: sources_lengths,
                                                                keep_prob_: 1.0})
                # update critic
                loss_critic, _ = critic_sess.run([l_, train_op_], {input_data_: source_batch,
                                         targets_: translate_logits,
                                         rewards_: rewards_c,
                                         lr_: learning_rate,
                                         target_sequence_length_: lens,
                                         source_sequence_length_: sources_lengths,
                                         keep_prob_: 1.0})

        # calculate td_error
        norm_rewards = (rewards_c - baselines)
        #update actor
        with train_sess.as_default():
            with train_sess.graph.as_default():
                lo, loss_actor, _ = train_sess.run(
                    [train_logits, cost, train_o],
                    {input_data: source_batch,
                     targets: translate_logits,
                     weights: norm_rewards,
                     lr: learning_rate,
                     target_sequence_length: lens,
                     source_sequence_length: sources_lengths,
                     keep_prob: 1.0})

        if batch_i % display_step == 0 and batch_i > 0:
            with train_sess.as_default():
                with train_sess.graph.as_default():
                    rewards_all = 0
                    for batch_j, (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths) in enumerate(
                            util.get_batches(valid_source, valid_target, batch_size,
                                             source_vocab_to_int['<PAD>'],
                                             target_vocab_to_int['<PAD>'])):
                        batch_valid_logits = train_sess.run(
                            inference_logits,
                            {input_data: valid_sources_batch,
                             source_sequence_length: valid_sources_lengths,
                             target_sequence_length: valid_targets_lengths,
                             keep_prob: 1.0})

                        rewards_v, pre_logits = util.get_bleu(valid_targets_batch, batch_valid_logits)
                        rewards_v = np.sum(rewards_v) / batch_size
                        rewards_all += rewards_v
                    rewards_all /= 10

            print('Epoch {:>3} Batch {:>4}/{} - Actor Loss: {:>6.4f} - Critic Loss: {:>6.4f} - Train Reward: {:>6.4f} - Valid Reward: {:>6.4f}'
                  .format(epoch_i, batch_i, len(source_int_text) // batch_size, loss_actor, loss_critic, reward_batch, rewards_all))

            loss_list.append((count, loss_actor))
            loss_list_critic.append((count, loss_critic))
            reward_list.append((count, reward_batch))
            reward_valid_list.append((count, rewards_all))

    with train_sess.as_default():
        with train_sess.graph.as_default():
            # Save Model
            saver_actor = tf.train.Saver()
            saver_actor.save(train_sess, save_path)

    with critic_sess.as_default():
        with critic_sess.graph.as_default():
            # Save Model
            saver_critic = tf.train.Saver()
            saver_critic.save(critic_sess, save_path_critic)

# plot
plot_loss(loss_list, "actor loss", "img/actor_loss.jpg")
plot_loss(loss_list_critic, "critic loss", "img/critic_loss.jpg")
plot_loss(reward_valid_list, "training reward", "img/training_reward.jpg")

#save model
util.save_params(save_path, 'params_actor_reinforce.p')
util.save_params(save_path_critic, 'params_critic_sup.p')
