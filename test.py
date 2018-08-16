import tensorflow as tf
import numpy as np
import util

# Batch Size
batch_size = 128

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = util.load_preprocess('preprocess.p')
load_path_sup = util.load_params('params_actor_sup.p')
load_path_actor = util.load_params('params_actor_reinforce.p')

source_test = util.read_list('source_test.npy')
target_test = util.read_list('target_test.npy')

test_acc_list = []

loaded_graph_sup = tf.Graph()
loaded_graph_actor = tf.Graph()
sup_sess = tf.Session(graph=loaded_graph_sup)
actor_sess = tf.Session(graph=loaded_graph_actor)
with sup_sess.as_default():
    with loaded_graph_sup.as_default():
        # Load saved model and restore the saved variables
        loader = tf.train.import_meta_graph(load_path_sup + '.meta')
        loader.restore(sup_sess, load_path_sup)

        input_data = loaded_graph_sup.get_tensor_by_name('input:0')
        logits = loaded_graph_sup.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph_sup.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = loaded_graph_sup.get_tensor_by_name('source_sequence_length:0')
        keep_prob = loaded_graph_sup.get_tensor_by_name('keep_prob:0')

        #
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                util.get_batches(source_test, target_test, batch_size,
                                 source_vocab_to_int['<PAD>'],
                                 target_vocab_to_int['<PAD>'])):
            translate_logits = sup_sess.run(logits, {input_data: source_batch,
                                                 target_sequence_length: targets_lengths,
                                                 source_sequence_length: sources_lengths,
                                                 keep_prob: 1.0})

            # print('logits:', translate_logits)
            # print('rnn_output:', translate_logits.rnn_output)

            print('Input')
            print('  Word Ids:      {}'.format([i for i in source_batch[0]]))
            print('  English Words: {}'.format(" ".join([source_int_to_vocab[i] for i in source_batch[0]
                                                         if i not in [0, 1]])))

            print('\nPrediction')
            print('  Word Ids:      {}'.format([i for i in translate_logits[0]]))
            print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits[0]
                                                        if i not in [0, 1]])))

            print('\nTarget')
            print('  Word Ids:      {}'.format([i for i in target_batch[0]]))
            print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in target_batch[0]
                                                        if i not in [0, 1]])))
            print("\n------------------------------------------------------------------\n")

            test_accs, predicts = util.get_bleu(target_batch, translate_logits)
            reward_batch = np.sum(test_accs)
            reward_avg = reward_batch / batch_size
            print('BLEU: ', reward_avg)
            test_acc_list.append(reward_avg)

        print("The BLEU score for the pretrain is {}".format(np.mean(test_acc_list)))

test_acc_list = []
with actor_sess.as_default():
    with loaded_graph_actor.as_default():
        # Load saved model and restore the saved variables
        loader = tf.train.import_meta_graph(load_path_actor + '.meta')
        loader.restore(actor_sess, load_path_actor)

        input_data = loaded_graph_actor.get_tensor_by_name('input:0')
        logits = loaded_graph_actor.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph_actor.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = loaded_graph_actor.get_tensor_by_name('source_sequence_length:0')
        keep_prob = loaded_graph_actor.get_tensor_by_name('keep_prob:0')

        #
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                util.get_batches(source_test, target_test, batch_size,
                                 source_vocab_to_int['<PAD>'],
                                 target_vocab_to_int['<PAD>'])):
            translate_logits = actor_sess.run(logits, {input_data: source_batch,
                                                     target_sequence_length: targets_lengths,
                                                     source_sequence_length: sources_lengths,
                                                     keep_prob: 1.0})

            # print('logits:', translate_logits)
            # print('rnn_output:', translate_logits.rnn_output)

            print('Input')
            print('  Word Ids:      {}'.format([i for i in source_batch[0]]))
            print('  English Words: {}'.format(" ".join([source_int_to_vocab[i] for i in source_batch[0]
                                                         if i not in [0, 1]])))

            print('\nPrediction')
            print('  Word Ids:      {}'.format([i for i in translate_logits[0]]))
            print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits[0]
                                                        if i not in [0, 1]])))

            print('\nTarget')
            print('  Word Ids:      {}'.format([i for i in target_batch[0]]))
            print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in target_batch[0]
                                                        if i not in [0, 1]])))
            print("\n------------------------------------------------------------------\n")

            test_accs, predicts = util.get_bleu(target_batch, translate_logits)
            reward_batch = np.sum(test_accs)
            reward_avg = reward_batch / batch_size
            print('BLEU: ', reward_avg)
            test_acc_list.append(reward_avg)

        print("The BLEU score for reinforcement is {}".format(np.mean(test_acc_list)))