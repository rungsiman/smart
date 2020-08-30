import tensorflow as tf
import torch


def describe_devices():
    # Get the GPU device name
    device_name = tf.test.gpu_device_name()

    # The device name should look like the following:
    if device_name == '/device:GPU:0':
        print('Found GPU at: {}'.format(device_name))
    else:
        print('GPU not found.')

    # If there’s a GPU available
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU
        device = torch.device('cuda:0')
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('Using GPU: ', torch.cuda.get_device_name(0))

    # If not, use cpu
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device('cpu')
