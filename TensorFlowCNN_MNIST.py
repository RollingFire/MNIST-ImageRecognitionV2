'''
By Austin Dorsey
Started: 9/11/18
Last modified: 9/22/18
Using the MNIST dataset and TensorFlow, created an image recognition Convolutional Neural Network that got an accuracy of 99.2%.
'''

import tensorflow as tf
import os
from datetime import datetime
import math


def getImgShifts(img, target, shift=2):
    '''Takes an tensor of images and the targets for the the images.
    Returns a tensor of 4 images per image that come from each image
    being shifted up, down, left, and right. Also returns a tensor
    of the matching targets.'''
    down = (tf.manip.roll(img, shift, axis=0))
    up = (tf.manip.roll(img, -shift, axis=0))
    right = (tf.manip.roll(img, shift, axis=1))
    left = (tf.manip.roll(img, -shift, axis=1))
    targetList = [target]
    return tf.concat([down, up, left, right], 0), tf.concat(targetList * 4, 0)


def stretchHeight(imgs, targets, stretch=1):
    '''Takes an tensor of images and the targets for the the images.
    Returns a tensor of the images that have been streaches taller
    and then resized to be the origenal shape. Also returns a tensor
    of the matching targets.'''
    shape = imgs.shape
    imgs = tf.image.resize_images(imgs, [shape[1] + (stretch * 2), shape[2]])
    imgs = tf.image.crop_to_bounding_box(imgs, stretch, 0, shape[1], shape[2])
    imgs = tf.cast(imgs, tf.float16)
    return imgs, targets


def scaleUp(imgs, targets, scale=1):
    '''Takes an tensor of images and the targets for the the images.
    Returns a tensor of the images that have been scaled bigger
    and then resized to be the origenal shape. Also returns a tensor
    of the matching targets.'''
    shape = imgs.shape
    imgs = tf.image.resize_images(imgs, [shape[1] + (scale * 2), shape[2] + (scale * 2)])
    imgs = tf.image.crop_to_bounding_box(imgs, scale, scale, shape[1], shape[2])
    imgs = tf.cast(imgs, tf.float16)
    return imgs, targets


def scaleDown(imgs, targets, scale=1):
    '''Takes an tensor of images and the targets for the the images.
    Returns a tensor of the images that have been scaled smaller
    and then resized to be the origenal shape. Also returns a tensor
    of the matching targets.'''
    shape = imgs.shape
    imgs = tf.image.crop_to_bounding_box(imgs, scale, scale, shape[1] - (scale * 2), shape[2] - (scale * 2))
    imgs = tf.image.resize_images(imgs, [shape[1], shape[2]])
    imgs = tf.cast(imgs, tf.float16)
    return imgs, targets


def miniBatches(x, y, batchSize=100):
    '''Takes two tensors of equal size and splits it into batches of batchSize.
    The last batch will be what is left less than batchSize.
    Returns lists of those batches.'''
    xBatch, yBatch = [], []
    a = int(x.shape[0])
    batches = math.ceil(a / batchSize)
    for batch in range(batches):
        xBatch.append(x[batchSize * batch : min(batchSize * (batch + 1), a)])
        yBatch.append(y[batchSize * batch : min(batchSize * (batch + 1), a)])
    return xBatch, yBatch


def evaluatingData():
    '''Construction phase for tensorflow that prepairs the data for evaluating models.
    Splits off 5000 images as a validation set.'''
    with tf.name_scope("MNIST"):
        global xTrain
        global yTrain
        global xVal
        global yVal
        mnist = tf.keras.datasets.mnist
        (xTrainTemp, yTrainTemp), (xTestTemp, yTestTemp) = mnist.load_data()
        xTrainTemp = tf.random_shuffle(xTrainTemp, seed=24)
        yTrainTemp = tf.random_shuffle(yTrainTemp, seed=24)
        xTrainTemp = tf.div(tf.cast(xTrainTemp, tf.float16), 255.0)
        xTrainTemp = tf.reshape(xTrainTemp, [int(xTrainTemp.shape[0]),28,28,1])
        xTrain = tf.identity(xTrainTemp[:55000], "xTrain")
        yTrain = tf.identity(yTrainTemp[:55000], "yTrain")
        xVal = tf.identity(xTrainTemp[55000:], "xVal")
        yVal = tf.identity(yTrainTemp[55000:], "yVal")
        xShifts, yShifts = getImgShifts(xTrain, yTrain, shift=1)
        xShifts2, yShifts2 = getImgShifts(xTrain, yTrain, shift=2)
        xStretch, yStretch = stretchHeight(xTrain, yTrain, stretch=1)
        xStretch2, yStretch2 = stretchHeight(xTrain, yTrain, stretch=2)
        xUp, yUp = scaleUp(xTrain, yTrain, scale=1)
        xDown, yDown = scaleDown(xTrain, yTrain, scale=1)
        xUp2, yUp2 = scaleUp(xTrain, yTrain, scale=2)
        xDown2, yDown2 = scaleDown(xTrain, yTrain, scale=2)
        xTrain = tf.concat([xTrain, xShifts, xShifts2, xStretch, xStretch2, xUp, xDown, xUp2, xDown2], 0)
        yTrain = tf.concat([yTrain, yShifts, yShifts2, yStretch, yStretch2, yUp, yDown, yUp2, yDown2], 0)


def fullTrainingData():
    '''Construction phase for tensorflow that prepairs the full dataset for training.'''
    with tf.name_scope("MNIST"):
        global xTrain
        global yTrain
        mnist = tf.keras.datasets.mnist
        (xTrainTemp, yTrainTemp), (xTestTemp, yTestTemp) = mnist.load_data()
        xTrain = tf.random_shuffle(xTrainTemp, seed=24)
        yTrain = tf.random_shuffle(yTrainTemp, seed=24)
        xTrain = tf.div(tf.cast(xTrain, tf.float16), 255.0)
        xTrain = tf.reshape(xTrain, [int(xTrainTemp.shape[0]),28,28,1])
        xTrain = tf.identity(xTrain, "xTrain")
        yTrain = tf.identity(yTrain, "yTrain")
        xShifts, yShifts = getImgShifts(xTrain, yTrain, shift=1)
        xShifts2, yShifts2 = getImgShifts(xTrain, yTrain, shift=2)
        xStretch, yStretch = stretchHeight(xTrain, yTrain, stretch=1)
        xStretch2, yStretch2 = stretchHeight(xTrain, yTrain, stretch=2)
        xUp, yUp = scaleUp(xTrain, yTrain, scale=1)
        xDown, yDown = scaleDown(xTrain, yTrain, scale=1)
        xUp2, yUp2 = scaleUp(xTrain, yTrain, scale=2)
        xDown2, yDown2 = scaleDown(xTrain, yTrain, scale=2)
        xTrain = tf.concat([xTrain, xShifts, xShifts2, xStretch, xStretch2, xUp, xDown, xUp2, xDown2], 0)
        yTrain = tf.concat([yTrain, yShifts, yShifts2, yStretch, yStretch2, yUp, yDown, yUp2, yDown2], 0)


def testData():
    '''Construction phase for tensorflow that prepairs the test set of images for testing the final model.'''
    with tf.name_scope("MNIST"):
        global xTest
        global yTest
        mnist = tf.keras.datasets.mnist
        (xTrainTemp, yTrainTemp), (xTestTemp, yTestTemp) = mnist.load_data()
        xTest = tf.div(tf.cast(xTestTemp, tf.float16), 255.0, name="xTest")
        xTest = tf.reshape(xTest, [int(xTest.shape[0]),28,28,1])
        yTest = tf.constant(yTestTemp, name="yTest")





with tf.name_scope("cnn"):
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="x")
    y = tf.placeholder(tf.int32, shape=(None), name="y")
    training = tf.placeholder(tf.bool, shape=(), name="training")
    learningRate = tf.placeholder(tf.float32, shape=(), name="learningRate")
    con1 = tf.layers.conv2d(x, filters=6, kernel_size=[5,5], strides=[1,1], padding="SAME")
    pool1 = tf.nn.max_pool(con1, ksize=[1,4,4,1], strides=[1,2,2,1], padding="VALID")
    con2 = tf.layers.conv2d(pool1, filters=4, kernel_size=[3,3], strides=[1,1], padding="SAME")
    con3 = tf.layers.conv2d(con2, filters=6, kernel_size=[3,3], strides=[1,1], padding="SAME")
    pool2 = tf.nn.max_pool(con3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
    dence = tf.layers.dense(tf.layers.flatten(pool2), 254, tf.nn.elu)
    denceDropout = tf.layers.dropout(dence, 0.6, training=training)
    dence2 = tf.layers.dense(denceDropout, 128, tf.nn.elu)
    denceDropout2 = tf.layers.dropout(dence2, 0.6, training=training)
    logits = tf.layers.dense(denceDropout2, 10)


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")


with tf.name_scope("train"):
    momentum = 0.9
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum=momentum, use_nesterov=True)
    trainingOp = optimizer.minimize(loss)


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("logging"):
    accSummary = tf.summary.scalar('ACC', accuracy)
    fileWriter = tf.summary.FileWriter(logdir, tf.get_default_graph())

folderPath = "Checkpoints"
saver = tf.train.Saver()


def trainAndEval():
    '''Uses the data construction created by evaluatingData. Trains the model on the training portion
    and tests it on the validation set. Prints the accuracy on the training data and the accuracy on
    the validation data. Logs data every 100 batches for tensorboard. Saves the final model to myModelEval.ckpt.'''
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rootLogdir = "tfLogs"
    logdir = "{}/run-{}/".format(rootLogdir, now)
    with tf.Session() as sess:
        epochs = 60
        batchSize = 200
        learningEpochs = [0, 30, 50]
        learningSchedule = [0.001, 0.0005, 0.0001]
        evaluatingData()
        xTrainBatches, yTrainBatches = miniBatches(xTrain, yTrain, batchSize=batchSize)
        init = tf.global_variables_initializer()
        sess.run(init)
        xTrainBatches = sess.run(xTrainBatches)
        yTrainBatches = sess.run(yTrainBatches)
        xEval = sess.run(xVal)
        yEval = sess.run(yVal)
        for epoch in range(epochs):
            if epoch in learningEpochs:
                rate = learningSchedule[learningEpochs.index(epoch)]
            for i, (xBatch, yBatch) in enumerate(zip(xTrainBatches, yTrainBatches)):
                sess.run(trainingOp, feed_dict={x: xBatch, y: yBatch, training: True, learningRate: rate})
                if i % 100 == 0:
                    summaryStr = accSummary.eval(feed_dict={x: xBatch, y: yBatch, training: False})
                    step = epoch * len(xTrainBatches) + i
                    fileWriter.add_summary(summaryStr, step)
            accTrain = accuracy.eval(feed_dict={x: xBatch, y: yBatch, training: False})
            accVal = accuracy.eval(feed_dict={x: xEval, y: yEval, training: False})
            print(epoch, "Train accuracy:", accTrain, "Val accuracy:", accVal)
        saver.save(sess, os.path.join(folderPath, "myModelEval.ckpt"))


def trainFinal():
    '''Uses the data construction created by fullTrainingData. Trains the model. Saves the final model to myModelEval.ckpt.'''
    with tf.Session() as sess:
        epochs = 60
        batchSize = 200
        learningEpochs = [0, 30, 50]
        learningSchedule = [0.001, 0.0005, 0.0001]
        fullTrainingData()
        xTrainBatches, yTrainBatches = miniBatches(xTrain, yTrain, batchSize=batchSize)
        init = tf.global_variables_initializer()
        sess.run(init)
        xTrainBatches = sess.run(xTrainBatches)
        yTrainBatches = sess.run(yTrainBatches)
        for epoch in range(epochs):
            print("Epoch", epoch)
            if epoch in learningEpochs:
                rate = learningSchedule[learningEpochs.index(epoch)]
            for i, (xBatch, yBatch) in enumerate(zip(xTrainBatches, yTrainBatches)):
                sess.run(trainingOp, feed_dict={x: xBatch, y: yBatch, training: True, learningRate: rate})
        saver.save(sess, os.path.join(folderPath, "myModelFinal.ckpt"))


def testFinal(file="myModelFinal.ckpt"):
    '''Tests the model of the file that is passed as a paramiter against the test data constructed by testData.
    Prints the accuracy of the model when finished.'''
    with tf.Session() as sess:
        testData()
        xTestData = sess.run(xTest)
        yTestData = sess.run(yTest)
        saver.restore(sess, os.path.join(folderPath, file))
        results = accuracy.eval(feed_dict={x: xTestData, y: yTestData, training: False})
        print("Accuracy of the model is:", (results * 100))
