import util
from distutils.version import LooseVersion
import tensorflow as tf
import warnings

source_path = 'data/small_vocab_en.txt'
target_path = 'data/small_vocab_fr.txt'

# 预处理数据，每个句子后面加上<EOS>
util.preprocess_and_save_data(source_path, target_path, 'preprocess.p')


# # Check TensorFlow Version
# assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
# print('TensorFlow Version: {}'.format(tf.__version__))
#
# # Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('No GPU found. Please use a GPU to train your neural network.')
# else:
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))