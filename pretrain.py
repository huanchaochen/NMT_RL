import util
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from seq2seq import seq2seq_model
from seq2seq import model_inputs
from plot import plot_loss
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


# Number of Epochs
epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 64
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 64
decoding_embedding_size = 64
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.8
# display step
display_step = 100


save_path = 'checkpoints/sup/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = util.load_preprocess('preprocess.p')
print('size:', len(target_vocab_to_int))#358
max_len = max([len(sentence) for sentence in target_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length, global_step = model_inputs()
    # td_error
    weights = tf.placeholder(tf.float32, shape=[None, None], name='weights')
    isSample = tf.placeholder(tf.bool, name='isTrain')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits, sample_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)

    # make a copy of train_logits
    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    training_samples = tf.identity(train_logits.sample_id)
    # make a copy of inference_logits
    inferencing_logits = tf.identity(inference_logits.sample_id, name='predictions')
    inferencing_output = tf.identity(inference_logits.rnn_output)
    # make a copy of sample_logits
    sampling_logits = tf.identity(sample_logits.sample_id, name='sample_logits')
    sampling_outputs = tf.identity(sample_logits.rnn_output, name='sample_outputs')

    # masks are created to facilitate the calculation of the loss.
    # We don't want to calculate the loss for the padding in a sentence.
    masks = tf.to_float(tf.greater(targets, 0))
    # number of words
    norm = tf.reduce_sum(masks)

    with tf.name_scope("optimization"):
        # get the probability distribution
        dist = tf.nn.softmax(training_logits)
        idx = tf.reshape(targets, [-1])
        idx_flattened = tf.range(0, tf.shape(dist)[1] * tf.shape(dist)[0]) * tf.shape(dist)[2] + idx
        # get the probability of target actions
        dist = tf.reshape(tf.gather(tf.reshape(dist, [-1]),
                      idx_flattened, [batch_size, -1]))
        log_dist = -tf.log(dist)
        masks = tf.multiply(masks, weights)
        cost = tf.reduce_sum(tf.multiply(log_dist, masks))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping is applied to mitigate the issue of exploding gradients
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step, name='train_op')


# split original data into training and test data
source_train, source_test, target_train, target_test = train_test_split(source_int_text,
                                                                        target_int_text, test_size=0.01, random_state=42)
# save dataset
util.save_list('source_train', source_train)
util.save_list('target_train', target_train)
util.save_list('source_test', source_test)
util.save_list('target_test', target_test)


print("The size for test data {},{}".format(len(source_test), len(target_test)))

# Split data to training and validation sets
valid_size = 1280
train_source = source_train[valid_size:]
train_target = target_train[valid_size:]
valid_source = source_train[:valid_size]
valid_target = target_train[:valid_size]

print("The size for training data {},{}".format(len(train_source), len(train_target)))
print("The size for validation data {},{}".format(len(valid_source), len(valid_target)))

loss_list = []
train_acc_list = []
valid_acc_list = []
count = 0

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('loss', cost)
    tf.add_to_collection('masks', masks)
    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                util.get_batches(train_source, train_target, batch_size,
                                 source_vocab_to_int['<PAD>'],
                                 target_vocab_to_int['<PAD>'])):
            max_len = np.max(targets_lengths)
            w = np.ones((batch_size, max_len), dtype=np.float32)
            count += 1
            training_logit, _, loss, m = sess.run(
                [train_logits, train_op, cost, masks],
                {input_data: source_batch,
                 targets: target_batch,
                 weights: w,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 max_target_sequence_length: max_len,
                 keep_prob: keep_probability
                 })

            loss_list.append((count, loss))

            if batch_i % display_step == 0 and batch_i > 0:
                rewards_all = 0
                for batch_j, (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths) in enumerate(
                        util.get_batches(valid_source, valid_target, batch_size,
                                         source_vocab_to_int['<PAD>'],
                                         target_vocab_to_int['<PAD>'])):
                    batch_valid_logits = sess.run(
                        inferencing_logits,
                        {input_data: valid_sources_batch,
                         source_sequence_length: valid_sources_lengths,
                         target_sequence_length: valid_targets_lengths,
                         keep_prob: 1.0})

                    rewards_v, pre_logits = util.get_bleu(valid_targets_batch, batch_valid_logits)
                    rewards_v = np.sum(rewards_v) / batch_size
                    rewards_all += rewards_v
                rewards_all /= 10

                print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.4f} - Valid bleu: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, loss, rewards_all))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')

plot_loss(loss_list, "pretrain loss", "img/pretrain_loss.jpg")

# Save parameters for checkpoint# Save p
util.save_params(save_path, 'params_actor_sup.p')
