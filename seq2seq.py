import tensorflow as tf
from tensorflow.python.layers.core import Dense


def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences for actor.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """

    # placeholder for input sequence data
    inputs = tf.placeholder(tf.int32, shape=[None, None], name="input")
    # placeholder for target sequence data
    targets = tf.placeholder(tf.int32, shape=[None, None], name="target")
    # placeholder for the learning rate of optimization process
    learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
    # placeholder for the keep probability of dropout
    keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob")
    # placeholder for lenght of the current target sequences
    target_sequence_length = tf.placeholder(tf.int32, shape=(None,), name="target_sequence_length")
    # the maximum length of the current target sequences
    max_target_len = tf.reduce_max(target_sequence_length, name='max_target_len')
    # placeholder for lenght of the current target sequences
    source_sequence_length = tf.placeholder(tf.int32, shape=(None,), name="source_sequence_length")
    # a variable for global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    return inputs, targets, learning_rate, keep_prob, target_sequence_length, max_target_len, source_sequence_length, global_step

def model_inputs_():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences for critic.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """

    # placeholder for input sequence data
    inputs = tf.placeholder(tf.int32, shape=[None, None], name="input_critic")
    # placeholder for target sequence data
    targets = tf.placeholder(tf.int32, shape=[None, None], name="target_critic")
    # placeholder for the learning rate of optimization process
    learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate_critic")
    # placeholder for the keep probability of dropout
    keep_prob = tf.placeholder(tf.float32, shape=[], name="keep_prob_critic")
    # placeholder for lenght of the current target sequences
    target_sequence_length = tf.placeholder(tf.int32, shape=(None,), name="target_sequence_length_critic")
    # the maximum length of the current target sequences
    max_target_len = tf.reduce_max(target_sequence_length, name='max_target_len_critic')
    # placeholder for lenght of the current target sequences
    source_sequence_length = tf.placeholder(tf.int32, shape=(None,), name="source_sequence_length_critic")
    # a variable for global step
    global_step = tf.Variable(0, name='global_step_critic', trainable=False)

    return inputs, targets, learning_rate, keep_prob, target_sequence_length, max_target_len, source_sequence_length, global_step


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for decoding
    return: Preprocessed target data
    """
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)

    return dec_input


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   source_sequence_length, source_vocab_size,
                   encoding_embedding_size):
    """
    Create encoding layer using lstm cell
    """

    # create the input sequence embedding
    embed_inputs = tf.contrib.layers.embed_sequence(rnn_inputs,
                                                    vocab_size=source_vocab_size,
                                                    embed_dim=encoding_embedding_size)

    # build lstm cell with dropout
    def build_lstm(num_units, keep_prob):
        lstm_cell = tf.contrib.rnn.LSTMCell(num_units, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

        dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                output_keep_prob=keep_prob)

        return dropout

    # build a multilayer LSTM cell
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [build_lstm(rnn_size, keep_prob) for _ in range(num_layers)])

    # Build a bidirectional RNN
    fw_cell = tf.contrib.rnn.MultiRNNCell(
        [build_lstm(rnn_size, keep_prob) for _ in range(num_layers)])
    bw_cell = tf.contrib.rnn.MultiRNNCell(
        [build_lstm(rnn_size, keep_prob) for _ in range(num_layers)])

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        embed_inputs,
        dtype=tf.float32,
        sequence_length=source_sequence_length)

    return tf.concat(bi_outputs, -1), bi_state[0]


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    """
    Create a decoding layer for training
    """

    helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                               sequence_length=target_sequence_length)

    decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                              helper=helper,
                                              initial_state=encoder_state,
                                              output_layer=output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_summary_length)

    return outputs


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for inference
    """

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=dec_embeddings,
                                                      start_tokens=tf.fill([batch_size], start_of_sequence_id),
                                                      end_token=end_of_sequence_id)

    decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                              helper=helper,
                                              initial_state=encoder_state,
                                              output_layer=output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_target_sequence_length)

    return outputs


def decoding_layer_sample(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for sample
    """

    helper = tf.contrib.seq2seq.SampleEmbeddingHelper(embedding=dec_embeddings,
                                                      start_tokens=tf.fill([batch_size], start_of_sequence_id),
                                                      end_token=end_of_sequence_id)

    decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                              helper=helper,
                                              initial_state=encoder_state,
                                              output_layer=output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_target_sequence_length)

    return outputs



def decoding_layer(dec_input, encoder_state, encoder_output,
                   source_sequence_length, target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer for actor
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size],
                                                   minval=-1,
                                                   maxval=1))

    dec_embed_inputs = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    def build_lstm(num_units, keep_prob):
        lstm_cell = tf.contrib.rnn.LSTMCell(num_units, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

        dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                output_keep_prob=keep_prob)

        return dropout

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [build_lstm(rnn_size, keep_prob) for _ in range(num_layers)])

    output_layer = Dense(target_vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                         use_bias=False)

    # Use attention mechanism for decoder
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units=rnn_size,
        memory=encoder_output,
        memory_sequence_length=source_sequence_length)

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(stacked_lstm,
                                                   attention_mechanism,
                                                   attention_layer_size=rnn_size)

    initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    initial_state = initial_state.clone(cell_state=encoder_state)

    with tf.variable_scope("decoder"):
        train_outputs = decoding_layer_train(initial_state, dec_cell, dec_embed_inputs,
                                             target_sequence_length, max_target_sequence_length,
                                             output_layer, keep_prob)

    with tf.variable_scope("decode", reuse=True):
        infer_outputs = decoding_layer_infer(initial_state, dec_cell, dec_embeddings, target_vocab_to_int['<GO>'],
                                             target_vocab_to_int['<EOS>'], max_target_sequence_length,
                                             target_vocab_size, output_layer, batch_size, keep_prob)

    with tf.variable_scope("sample", reuse=True):
        sample_outputs = decoding_layer_sample(initial_state, dec_cell, dec_embeddings, target_vocab_to_int['<GO>'],
                                             target_vocab_to_int['<EOS>'], max_target_sequence_length,
                                             target_vocab_size, output_layer, batch_size, keep_prob)

    return train_outputs, infer_outputs, sample_outputs


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    encoder_output, encoder_state = encoding_layer(input_data, rnn_size, num_layers, keep_prob,
                                                   source_sequence_length, source_vocab_size,
                                                   enc_embedding_size)

    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)

    train_outputs, infer_outputs, sample_outputs = decoding_layer(dec_input, encoder_state, encoder_output,
                                                  source_sequence_length, target_sequence_length, max_target_sentence_length,
                                                  rnn_size,
                                                  num_layers, target_vocab_to_int, target_vocab_size,
                                                  batch_size, keep_prob, dec_embedding_size)

    return train_outputs, infer_outputs, sample_outputs


def decoding_layer_critic(dec_input, encoder_state, encoder_output,
                   source_sequence_length, target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer for critic
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size],
                                                   minval=-1,
                                                   maxval=1))

    dec_embed_inputs = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    def build_lstm(num_units, keep_prob):
        lstm_cell = tf.contrib.rnn.LSTMCell(num_units, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

        dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                output_keep_prob=keep_prob)

        return dropout

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [build_lstm(rnn_size, keep_prob) for _ in range(num_layers)])

    # Use attention mechanism for decoder
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units=rnn_size,
        memory=encoder_output,
        memory_sequence_length=source_sequence_length)

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(stacked_lstm,
                                                   attention_mechanism,
                                                   attention_layer_size=rnn_size)

    initial_state = dec_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    initial_state = initial_state.clone(cell_state=encoder_state)

    with tf.variable_scope("decoder"):
        train_outputs = decoding_layer_train(initial_state, dec_cell, dec_embed_inputs,
                                             target_sequence_length, max_target_sequence_length,
                                             None, keep_prob)

    return train_outputs


def critic_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network for critic
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    encoder_output, encoder_state = encoding_layer(input_data, rnn_size, num_layers, keep_prob,
                                                   source_sequence_length, source_vocab_size,
                                                   enc_embedding_size)

    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)

    train_outputs = decoding_layer_critic(dec_input, encoder_state, encoder_output,
                                                  source_sequence_length, target_sequence_length, max_target_sentence_length,
                                                  rnn_size,
                                                  num_layers, target_vocab_to_int, target_vocab_size,
                                                  batch_size, keep_prob, dec_embedding_size)

    return train_outputs
