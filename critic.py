import tensorflow as tf
import numpy as np
import util
from seq2seq import critic_model
from seq2seq import model_inputs_
from plot import plot_loss

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
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.8
save_path_critic = 'checkpoints/sup/critic/dev'

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = util.load_preprocess('preprocess.p')
load_path_actor = util.load_params('params_actor_sup.p')

source_train = util.read_list('source_train.npy')
target_train = util.read_list('target_train.npy')

test_acc_list = []

train_graph = tf.Graph()
critic_graph = tf.Graph()
actor_graph = tf.Graph()

train_sess = tf.Session(graph=train_graph)
critic_sess = tf.Session(graph=critic_graph)
actor_sess = tf.Session(graph=actor_graph)



with train_sess.as_default():
    with train_graph.as_default():
        # Load saved model and restore the saved variables
        loader = tf.train.import_meta_graph(load_path_actor + '.meta')
        loader.restore(train_sess, load_path_actor)

        input_data = train_graph.get_tensor_by_name('input:0')
        logits = train_graph.get_tensor_by_name('sample_logits:0')
        # train_logits = train_graph.get_tensor_by_name('logits:0')
        target_sequence_length = train_graph.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = train_graph.get_tensor_by_name('source_sequence_length:0')
        keep_prob = train_graph.get_tensor_by_name('keep_prob:0')
        weights = train_graph.get_tensor_by_name('weights:0')
        targets = train_graph.get_tensor_by_name('target:0')
        # train_o = train_graph.get_tensor_by_name('train_op:0')
        # cost = train_graph.get_tensor_by_name('loss:0')
        lr = train_graph.get_tensor_by_name('learning_rate:0')
        global_step = train_graph.get_tensor_by_name('global_step:0')
        max_target_sequence_length = train_graph.get_tensor_by_name('max_target_len:0')
        train_o = tf.get_collection('train_op')[0]
        cost = tf.get_collection('loss')[0]


with critic_sess.as_default():
    with critic_graph.as_default():
        input_data_, targets_, lr_, keep_prob_, target_sequence_length_, max_target_sequence_length_, source_sequence_length_, global_step_ = model_inputs_()
        rewards_ = tf.placeholder(tf.float32, shape=[None, None], name="rewards")
        # samples_ = tf.placeholder(tf.int32, shape=[None, None], name="samples_")
        input_shape = tf.shape(input_data)
        # r = tf.Variable([rewards_] * target_sequence_length_)
        W = tf.Variable(tf.zeros([64, 1]), name="weight", dtype=tf.float32)
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

        # make a copy of train_logits
        training_logits_ = tf.identity(train_logits_.rnn_output)
        training_logits_ = tf.squeeze(tf.matmul(tf.reshape(training_logits_, [-1, 64]), W))
        training_logits_ = tf.reshape(training_logits_, [batch_size, -1], name='logits_critic')
        # inference_logits = tf.identity(tf.squeeze(inference_logits.rnn_output), name='predictions_critic')
        # training_samples_ = tf.identity(train_logits.sample_id)
        # training_logits_ = tf.clip_by_value(training_logits_, 0, 100)

        # inference_logits_ = tf.identity(tf.squeeze(inference_logits.rnn_output), name='logits_critic')
        # masks_ = tf.sequence_mask(target_sequence_length_, max_target_sequence_length_, dtype=tf.float32)
        masks_ = tf.to_float(tf.greater(targets_, 0))
        norm = tf.reduce_sum(masks_)
        with tf.name_scope("optimization_critic"):
            # Loss function
            # l_ = tf.losses.mean_squared_error(inference_logits_, rewards_, masks_)
            l_ = tf.reduce_sum(tf.multiply(tf.squared_difference(training_logits_, rewards_), masks_)) / norm

            # Optimizer
            optimizer_ = tf.train.AdamOptimizer(lr_)

            # Gradient Clipping is applied to mitigate the issue of exploding gradients
            gradients_ = optimizer_.compute_gradients(l_)
            capped_gradients_ = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients_ if grad is not None]
            train_op_ = optimizer_.apply_gradients(capped_gradients_, global_step=global_step_)

with critic_sess.as_default():
    with critic_sess.graph.as_default():
        critic_sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        tf.add_to_collection('train_op_critic', train_op_)
        tf.add_to_collection('loss_critic', l_)

loss_list = []
count = 0
for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
            util.get_batches(source_train, target_train, batch_size,
                             source_vocab_to_int['<PAD>'],
                             target_vocab_to_int['<PAD>'])):
    count += 1
    with train_sess.as_default():
        with train_sess.graph.as_default():

            translate_logits = train_sess.run(logits, {input_data: source_batch,
                                             target_sequence_length: targets_lengths,
                                             source_sequence_length: sources_lengths,
                                             keep_prob: 1.0})

    lens = [[translate_logits.shape[1]] * batch_size]
    lens = np.squeeze(lens)
    # print(lens)
    rewards, pre_logits = util.get_bleu(target_batch, translate_logits)
    # print(targets_lengths[0])
    rewards_c = [[rewards[0]] * lens[0]]
    # print(rewards_c)
    for i in range(batch_size - 1):
        temp = [[rewards[i + 1]] * lens[i + 1]]
        # print(temp)
        rewards_c = np.vstack((rewards_c, temp))

    print('reward:', rewards_c.shape)

    # print(len(rewards))
    reward_batch = np.sum(rewards)
    with critic_sess.as_default():
        with critic_sess.graph.as_default():
            # critic_sess.run(tf.global_variables_initializer())

            baselines, _, loss_, m = critic_sess.run(
                [training_logits_, train_op_, l_, masks_], {input_data_: target_batch,
                                     targets_: translate_logits,
                                     rewards_: rewards_c,
                                     lr_: learning_rate,
                                     target_sequence_length_: lens,
                                     source_sequence_length_: targets_lengths,
                                     keep_prob_: keep_probability})
            # print(m)
            print('baseline:', baselines)
            print(baselines.shape)
            print('critic loss:', loss_)
            # print(baselines.shape)
            loss_list.append((count, loss_))
            saver.save(critic_sess, save_path_critic)
            print('Model Trained and Saved')

            util.save_params(save_path_critic, 'params_critic_sup.p')
plot_loss(loss_list)


