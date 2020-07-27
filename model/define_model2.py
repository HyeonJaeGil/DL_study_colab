import time
from abc import abstractmethod
import tensorflow as tf
import numpy as np
from model.layers import conv_layer, maxpool_layer, fc_layer
import matplotlib.pyplot as plt


def get_tf_variable(shape, name):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1), name=name, trainable=True, dtype=tf.float32)


class ConvNet(object):
    """Base class for Convolutional Neural Networks."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        Model initializer.
        :param input_shape: tuple, the shape of inputs (H, W, C), ranged [0.0, 1.0].
        :param num_classes: int, the number of classes.
        """

        # Build model and loss function
        self.d = self._build_model(**kwargs)
        self.logits = self.d['logits']
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)

    @abstractmethod
    def _build_model(self, **kwargs):
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        pass

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        Make predictions for the given dataset.
        :param sess: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, whether to print details during prediction.
        :param kwargs: dict, extra arguments for prediction.
            - batch_size: int, batch size for each iteration.
            - augment_pred: bool, whether to perform augmentation for prediction.
        :return _y_pred: np.ndarray, shape: (N, num_classes).
        """
        batch_size = kwargs.pop('batch_size', 256)
        augment_pred = kwargs.pop('augment_pred', True)

        if dataset.labels is not None:
            assert len(dataset.labels.shape) > 1, 'Labels must be one-hot encoded.'
        num_classes = int(self.y.get_shape()[-1])
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size

        if verbose:
            print('Running prediction loop...')

        # Start prediction loop
        _y_pred = []
        start_time = time.time()
        for i in range(num_steps+1):
            if i == num_steps:
                _batch_size = pred_size - num_steps*batch_size
            else:
                _batch_size = batch_size
            X, _ = dataset.next_batch(_batch_size, shuffle=False,
                                      augment=augment_pred, is_train=False)
            # if augment_pred == True:  X.shape: (N, 10, h, w, C)
            # else:                     X.shape: (N, h, w, C)

            # If performing augmentation during prediction,
            if augment_pred:
                y_pred_patches = np.empty((_batch_size, 10, num_classes),
                                          dtype=np.float32)    # (N, 10, num_classes)
                # compute predictions for each of 10 patch modes,
                for idx in range(10):
                    y_pred_patch = sess.run(self.pred,
                                            feed_dict={self.X: X[:, idx],    # (N, h, w, C)
                                                       self.is_train: False})
                    y_pred_patches[:, idx] = y_pred_patch
                # and average predictions on the 10 patches
                y_pred = y_pred_patches.mean(axis=1)    # (N, num_classes)
            else:
                # Compute predictions
                y_pred = sess.run(self.pred,
                                  feed_dict={self.X: X,
                                             self.is_train: False})    # (N, num_classes)

            _y_pred.append(y_pred)
        if verbose:
            print('Total prediction time(sec): {}'.format(time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)    # (N, num_classes)

        return _y_pred


class MyModel():

    def __init__(self, x_input, y_input, **kwargs):

        # Build model and loss function
        self.d = self._build_model(x_input, **kwargs)
        self.loss = self._build_loss(y_input, **kwargs)

        # self.pool_size = 3
        self.dropout = kwargs.pop('dropout_prob', 0.0)
        self.n_classes = kwargs.pop('n_classes', 1000)

        self.shapes = [
            [11, 11, 3, 96],
            [5, 5, 96, 256],
            [3, 3, 256, 384],
            [3, 3, 384, 384],
            [3, 3, 384, 384],
            [6 * 6 * 256, 4096],
            [4096, 4096],
            [4096, self.n_classes]
        ]

        self.weights = []
        for i in range(len(self.shapes)):
            self.weights.append(get_tf_variable(self.shapes[i], 'weight{}'.format(i)))

        self.bias = []
        for i in range(len(self.shapes)):
            self.bias.append(get_tf_variable([1, self.shapes[i][-1]], 'bias{}'.format(i)))

    def _build_model(self, x_input, **kwargs):

        conv1 = conv_layer(x_input, self.weights[0], self.bias[0])
        pool1 = maxpool_layer(conv1, poolSize=3, stride=2)

        conv2 = conv_layer(pool1, self.weights[1], self.bias[1])
        pool2 = maxpool_layer(conv2, poolSize=3, stride=2)

        conv3 = conv_layer(pool2, self.weights[2], self.bias[2])
        conv4 = conv_layer(conv3, self.weights[3], self.bias[3])
        conv5 = conv_layer(conv4, self.weights[4], self.bias[4])

        flat1 = tf.reshape(conv5, [-1, conv5.shape[1] * conv5.shape[2] * conv5.shape[3]])

        fully1 = tf.nn.relu(fc_layer(flat1, self.weights[5], self.bias[5]))
        fully1_dropout = tf.nn.dropout(fully1, rate=self.dropout)

        fully2 = tf.nn.relu(fc_layer(fully1_dropout, self.weights[6], self.bias[6]))
        fully2_dropout = tf.nn.dropout(fully2, rate=self.dropout)

        y_predict = fc_layer(fully2_dropout, self.weights[7], self.bias[7])

        # print(conv1.shape,pool1.shape,conv2.shape,pool2.shape,flat1.shape,fully1.shape,y_pred.shape)

        return y_predict

    def trainable_variables(self):

        return self.weights + self.bias

    def _build_loss(self, y_input, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments for regularization term.
            - weight_decay: float, L2 weight decay regularization coefficient.
        :return tf.Tensor.
        """

        weight_decay = kwargs.pop('weight_decay', 0.0005)
        variables = self.trainable_variables()
        l2_reg_loss = tf.add_n([tf.nn.l2_loss(var) for var in variables])

        # Softmax cross-entropy loss function
        (y_pred, y_true) = y_input
        softmax_losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_true),
                                                                 logits=y_pred)
        softmax_loss = tf.reduce_mean(input_tensor=softmax_losses)

        return softmax_loss + weight_decay * l2_reg_loss

    def train_step(self, x_input, y_true, epoch):
        epoch_accuracy = None
        epoch_loss_avg = None

        with tf.GradientTape() as tape:
            # Get the predictions
            preds = model.run(x_input)

            # Calc the loss
            current_loss = self._build_loss(preds, y_true)

            # Get the gradients
            grads = tape.gradient(current_loss, model.trainable_variables())

            # Update the weights
            optimizer.apply_gradients(zip(grads, model.trainable_variables()))

            if epoch % 100 == 0:
                y_pred = model.run(ch.test_images)
                matches = tf.equal(tf.math.argmax(y_pred, 1), tf.math.argmax(ch.test_labels, 1))

                epoch_accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))
                epoch_loss_avg = tf.reduce_mean(current_loss)

                print("--- On epoch {} ---".format(epoch))
                tf.print("Accuracy: ", epoch_accuracy, "| Loss: ", epoch_loss_avg)
                print("\n")

            return epoch_accuracy, epoch_loss_avg