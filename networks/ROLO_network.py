import cv2
import tensorflow as tf
import numpy as np
from utils import ROLO_utils as utils

class ROLO_TF:
    disp_console = True

    # YOLO parameters
    w_img, h_img = [352, 240]

    # ROLO Network Parameters
    weights_file = 'weights/model_step3_exp2.ckpt'
    num_steps = 3  # number of frames as an input sequence
    num_feat = 4096
    num_predict = 6 # final output of LSTM 6 loc parameters
    num_input = num_feat + num_predict # data input: 4096+6= 5002

    # ROLO Parameters
    batch_size = 1

    def __init__(self,argvs = []):
        self.build_networks()

    def build_networks(self):
        if self.disp_console : print "Building ROLO graph..."

        graph = tf.Graph()
        with graph.as_default():
            self.x = tf.placeholder("float32", [None, self.num_steps, self.num_input])
            self.istate = tf.placeholder("float32", [None, 2 * self.num_input]) #state & cell => 2x num_input
            # input shape: (batch_size, n_steps, n_input)
            _X = tf.transpose(self.x, [1, 0, 2])  # permute num_steps and batch_size
            # Reshape to prepare input to hidden activation
            _X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input]) # (num_steps*batch_size, num_input)
            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            _X = tf.split(0, self.num_steps, _X) # n_steps * (batch_size, num_input)

            cell = tf.nn.rnn_cell.LSTMCell(self.num_input, self.num_input)
            state = self.istate
            for step in range(self.num_steps):
                outputs, state = tf.nn.rnn(cell, [_X[step]], state)
                tf.get_variable_scope().reuse_variables()

            self.pred_location = outputs[0][:, 4097:4101]

            self.sess = tf.Session(graph = graph)
            saver = tf.train.Saver()
            saver.restore(self.sess, self.weights_file)

        if self.disp_console : print "Loading complete!" + '\n'

    def track(self, yolo_outputs, img):
        batch_xs = np.reshape(yolo_outputs, [self.batch_size, self.num_steps, self.num_input])
        istate = np.zeros((self.batch_size, 2 * self.num_input))

        rolo_location = self.sess.run(
                self.pred_location,
                feed_dict = { self.x: batch_xs, self.istate: istate })

        height, width, channels = img.shape
        normal_rolo_location = utils.locations_normal(width, height, rolo_location[0])
        x = int(normal_rolo_location[0])
        y = int(normal_rolo_location[1])
        w = int(normal_rolo_location[2])
        h = int(normal_rolo_location[3])
        pred_location = [x- w // 2, y - h // 2, w, h]

        return pred_location
