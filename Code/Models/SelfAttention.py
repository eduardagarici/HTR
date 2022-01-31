from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

class Model:

    def __init__(self,imageSize,charList, maxTextLength,batchSize, mustRestore = False):
        self.charList = charList
        self.imageSize = imageSize
        self.batchSize = batchSize
        self.maxTextLength = maxTextLength
        self.snapID = 0
        self.mustRestore = mustRestore

        # CNN
        with tf.name_scope('CNN'):
            with tf.name_scope('Input'):
                self.inputImages = tf.placeholder(tf.float32, shape=(
                    self.batchSize, imageSize[0], imageSize[1]))
            cnnOut4d = self.setupCNN(self.inputImages)

        # RNN

        with tf.name_scope('RNN'):
            rnnOut3d = self.setupRNN(cnnOut4d)

        # Debuging CTC
        self.rnnOutput = tf.nn.softmax(tf.transpose(rnnOut3d, [1, 0, 2]))

        # CTC
        with tf.name_scope('CTC'):
            (self.loss, self.decoder) = self.setupCTC(rnnOut3d)

        # Optimize NN parameters
        with tf.name_scope('Optimizer'):
            self.batchesTrained = 0
            self.learningRate = tf.placeholder(tf.float32, shape=[])
            self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).\
                minimize(self.loss)

        # Initializer

        (self.sess, self.saver) = self.setupTF()

    def setupCNN(self, cnnIn3d):

        cnnIn4d = tf.expand_dims(input=cnnIn3d, axis = 3)

        # First Layer: Conv(5x5) + Pool (2x2) - Output size : 400 x 32 x 64
        with tf.name_scope('Conv_Pool_1'):

            conv = tf.keras.layers.Conv2D(filters = 64, kernel_size= 5, padding='SAME', activation='relu')(cnnIn4d)
            pool = tf.keras.layers.MaxPool2D(pool_size =(2,2), padding = 'VALID')(conv)

        # Second Layer: Conv (5x5) + Pool(1x2) - Output size: 400 x 16 x 128
        with tf.name_scope('Conv_Pool_2'):

            conv = tf.keras.layers.Conv2D(filters = 128, kernel_size= 5, padding='SAME', activation='relu')(pool)
            pool = tf.keras.layers.MaxPool2D(pool_size = (1,2), padding='VALID')(conv)

        # Third Layer: Conv (3x3) + Pool (2x2) + Simple Batch Norm - Output size: 200 x 8 x 128
        with tf.name_scope('Conv_Pool_BN_3'):

            conv = tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding='SAME')(pool)
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            relu = tf.keras.layers.ReLU()(batch_norm)
            pool = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='VALID')(relu)

        # Fourth Layer: Conv (3x3) - Output size: 200 x 8 x 256
        with tf.name_scope('Conv_4'):

            conv = tf.keras.layers.Conv2D(filters = 256, kernel_size= 3, padding='SAME', activation='relu')(pool)

        # Fifth Layer: Conv (3x3) + Pool(2x2) - Output size: 100 x 4 x 256
        with tf.name_scope('Conv_Pool_5'):

            conv = tf.keras.layers.Conv2D(filters = 256, kernel_size = 3 , padding ='SAME', activation='relu')(conv)
            pool = tf.keras.layers.MaxPool2D(pool_size=(2,2), padding = 'VALID')(conv)

        # Sixth Layer: Conv (3x3) + Pool(1,2) + Simple Batch Norm - Output size: 100 x 2 x 512
        with tf.name_scope('Conv_Pool_BN_6'):

            conv = tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, padding='SAME')(pool)
            mean, variance = tf.nn.moments(conv, axes=[0])
            batch_norm = tf.nn.batch_normalization(
                conv, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
            relu = tf.keras.layers.ReLU()(batch_norm)
            pool = tf.keras.layers.MaxPool2D(pool_size = (1,2), padding='VALID')(relu)

        # Seventh Layer: Conv (3x3) + Pool (1x2) - Output size: 100 x 1 x 512
        with tf.name_scope('Conv_Pool_7'):

            conv = tf.keras.layers.Conv2D(filters = 512, kernel_size=3, padding='SAME', activation='relu')(pool)
            pool = tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='VALID')(conv)

        return pool

    def setupRNN(self, rnnIn4d):

        rnnIn4d = tf.slice(rnnIn4d, [0, 0, 0, 0], [self.batchSize, 100, 1, 216])
        rnnIn3d = tf.squeeze(rnnIn4d)

        # 2 layers of LSTM cell used to build RNN
        numHidden = 216
        bilstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(numHidden, return_sequences=True), merge_mode="concat")(rnnIn3d)
        bilstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(numHidden, return_sequences=True), merge_mode="concat")(bilstm1)
        Q = tf.keras.layers.Dense(2 * numHidden)(bilstm2)
        K = tf.keras.layers.Dense(2 * numHidden)(bilstm2)
        V = tf.keras.layers.Dense(len(self.charList) + 1)(bilstm2)

        attention = self.attention_fun(Q,K)
        output = tf.matmul(attention, V)
        print(output.shape)
        return output
    
    def setupCTC(self, ctcIn3d):

        ctcIn3dTBC = tf.transpose(ctcIn3d, [1, 0, 2])

        # Ground truth text as sparse tensor

        with tf.name_scope('CTC_Loss'):
            self.groundTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[
                None,2]), tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

            # Calculate loss for batch
            self.sequenceLength = tf.placeholder(tf.int32, [None])
            loss = tf.nn.ctc_loss(labels = self.groundTexts, inputs = ctcIn3dTBC, sequence_length= self.sequenceLength,
                                  ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=True)

        with tf.name_scope('CTC_Decoder'):
            decoder = tf.nn.ctc_beam_search_decoder(inputs = ctcIn3dTBC, sequence_length=self.sequenceLength)
            #decoder =  word_beam_search()

        return (tf.reduce_mean(loss), decoder)

    def attention_fun(self,Q, K, scaled_=True, masked_=False):
        attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

        if scaled_:
            d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
            attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

        if masked_:
            raise NotImplementedError

        attention = tf.nn.softmax(attention, dim=-1)  # [batch_size, sequence_length, sequence_length]
        return attention

    def input_fun(**config):
        data = tf.random_normal((
            config['batch_size'], config['sequence_length'], config['hidden_dim']))
        return data

    def setupTF(self):

        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep = 1)
        modelDir = 'drive/My Drive/Licenta/Cod/Checkpoints/IAM/SelfAttention/'
        latestSnapshot = tf.train.latest_checkpoint(modelDir)

        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in ' + modelDir)

        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())

        return (sess, saver)

    def toSparse(self, texts):
        """ Convert ground truth texts into sparse tensor for ctc_loss """
        indices = []
        values = []
        shape = [len(texts), 0]  # Last entry must be max(labelList[i])
        # Go over all texts
        for (batchElement, texts) in enumerate(texts):
            # Convert to string of label (i.e. class-ids)
            labelStr = []
            for c in texts:
                labelStr.append(self.charList.index(c))
            # labelStr = [self.charList.index(c) for c in texts]
            # Sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # Put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput):
        """ Extract texts from output of CTC decoder """
        # Contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(self.batchSize)]
        # Word beam search: label strings terminated by blank

        # Ctc returns tuple, first element is SparseTensor
        decoded = ctcOutput[0][0]
        # Go over all indices and save mapping: batch -> values
        # idxDict = {b : [] for b in range(Model.batchSize)}
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batchElement = idx2d[0]  # index according to [b,t]
            encodedLabelStrs[batchElement].append(label)
        # Map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def trainBatch(self, batch):
        """ Feed a batch into the NN to train it """
        sparse = self.toSparse(batch.groundTexts)
        rate = 0.01 if self.batchesTrained < 10 else (
            0.001 if self.batchesTrained < 2750 else 0.0001)
        if self.mustRestore:
        	rate = 0.0001
        (_, lossVal) = self.sess.run([self.optimizer, self.loss], {
            self.inputImages: batch.images, self.groundTexts: sparse,
            self.sequenceLength: [self.maxTextLength] * self.batchSize,
            self.learningRate: rate})

        self.batchesTrained += 1
        return lossVal

    def inferBatch(self, batch):


        decoded, rnnOutput = self.sess.run([self.decoder, self.rnnOutput], {
                                            self.inputImages: batch.images,
                                            self.sequenceLength: [self.maxTextLength] * self.batchSize})

        return self.decoderOutputToText(decoded), rnnOutput

    def save(self):
        """ Save model to file """
        self.snapID += 1
        self.saver.save(self.sess, 'drive/My Drive/Licenta/Cod/Checkpoints/IAM/SelfAttention/',
                        global_step=self.snapID)