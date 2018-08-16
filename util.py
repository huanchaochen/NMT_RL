import os
import pickle
import copy
import nltk
import numpy as np
import tensorflow as tf

CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to corresponding word ids
    """
    # convert source text to the corresponding id
    source_id_text = []
    for text in source_text.split('\n'):
        source_id_text.append([source_vocab_to_int.get(word, source_vocab_to_int['<UNK>']) for word in text.split()])
    
    # convert source text to the corresponding id
    target_id_text = []
    for text in target_text.split('\n'):
        target_id_text.append([target_vocab_to_int.get(word, target_vocab_to_int['<UNK>']) for word in text.split()] \
                              + [target_vocab_to_int['<EOS>']])
    
    return source_id_text, target_id_text

# def remove_lines(path):
#     input_file = os.path.join(path)
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = []
#         line = f.readline()
#         while line:
#             line = f.readline()
#             if(len(line) <= 50):
#                 line = line.rstrip()
#                 data.append(line)
#     with open('', 'w') as f:

def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_and_save_data(source_path, target_path, save_path):
    # Preprocess Text Data.  Save to to file.
    
    # Preprocess
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    source_text = source_text.lower()
    target_text = target_text.lower()

    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    # Save Data
    # with open('preprocess.p', 'wb') as out_file:
    with open(save_path, 'wb') as out_file:
        pickle.dump((
            (source_text, target_text),
            (source_vocab_to_int, target_vocab_to_int),
            (source_int_to_vocab, target_int_to_vocab)), out_file)


def preprocess(source_path, target_path, source_vocab_to_int, target_vocab_to_int):
    # Preprocess Text Data.

    # Preprocess
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    source_text = source_text.lower()
    target_text = target_text.lower()

    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    return source_text, target_text

def load_preprocess(data_path):
    
    #Load the Preprocessed Training data and return them in batches of <batch_size> or less
    
    # with open('preprocess.p', mode='rb') as in_file:
    with open(data_path, mode='rb') as in_file:
        return pickle.load(in_file)


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    """
    vocab = set(text.split())
    vocab_to_int = copy.copy(CODES)

    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i

    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


def save_params(params, path):
    """
    Save parameters to file
    """
    with open(path, 'wb') as out_file:
        pickle.dump(params, out_file)


def load_params(path):
    """
    Load parameters from file
    """
    with open(path, mode='rb') as in_file:
        return pickle.load(in_file)

def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


def get_bleu(target, logits):
    """
    Calculate the bleu score
    """
    bleu_scores = [nltk.translate.bleu_score.sentence_bleu([ref], hypo, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4) * 100 for ref, hypo in zip(target, logits)]
    # return np.mean(bleu_scores)
    return bleu_scores, logits


def save_list(path, list_data):
    n_data = np.array(list_data)
    np.save(path, n_data)


def read_list(path):
    list_read = np.load(path).tolist()
    return list_read


def get_pro(x, idx):
    idx = tf.reshape(idx, [-1])
    idx_flattened = tf.range(0, x.shape[1] * x.shape[0]) * x.shape[2] + idx
    y = tf.gather(tf.reshape(x, [-1]),  # flatten input
                  idx_flattened)  # use flattened indices
    # x = tf.reshape(x, [-1])
    y = tf.reshape(y, [x.shape[0], -1])
    return y


