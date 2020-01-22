import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

import constants
import helper_methods

tensorBoardPath = '/TensorBoard_models/'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


imgWidth, imgHeight, imgDepth = constants.IMGSIZE, constants.IMGSIZE, constants.IMGDEPTH

if len(sys.argv) < 2:
    print("Too few arguments provided, program terminating")
    sys.exit(0)

#data
collectiveData = tf.placeholder('float', [None, imgWidth, imgHeight, imgDepth])
ballData = tf.placeholder('float',[None, imgWidth, imgHeight, imgDepth])
bottleData = tf.placeholder('float',[None, imgWidth, imgHeight, imgDepth])
canData = tf.placeholder('float',[None, imgWidth, imgHeight, imgDepth])
paperData = tf.placeholder('float',[None, imgWidth, imgHeight, imgDepth])
backgroundData = tf.placeholder('float',[None, imgWidth, imgHeight, imgDepth])

#labels
collectiveLabels = tf.placeholder('float', [None, constants.CLASSCOUNT])
ballLabels = tf.placeholder('float',[None, 1])
bottleLabels = tf.placeholder('float',[None, 1])
canLabels = tf.placeholder('float',[None, 1])
paperLabels = tf.placeholder('float',[None, 1])
backgroundLabels = tf.placeholder('float',[None, 1])

#Define the 1D array that will hold tensor for the image
#NOTE: Need to change so that it can take constants.CNN_LAYER instead of image depth
tensorShape = int(imgWidth * imgHeight * imgDepth)
# print(tensorShape)

weights, bias = {}, {}

#Now we can create the individual CNN's for each object in the data set

#Ball CNN
weights['ballWeight'] = tf.Variable(tf.random_normal([7,7,3,constants.FILTERCOUNT])) #This is the 'kernel' variable, but unlike keras it takes a matrix (height, width, in-channels, out-channels)
bias['ballBias'] = tf.Variable(tf.random_normal([constants.FILTERCOUNT]))
ballConv = tf.nn.conv2d(input=ballData, filter=weights['ballWeight'], strides=[1,1,1,1], padding=constants.PADDING)
ballActivation = tf.nn.relu(ballConv + bias['ballBias'])
weights['ballOutputWeight'] = tf.Variable(tf.random_normal([tensorShape, 1]))
bias['ballOutputBias'] = tf.Variable(tf.random_normal([1]))
ballOutput = tf.reshape(ballActivation, shape=[-1, tensorShape])
ballPrediction = tf.matmul(ballOutput, weights['ballOutputWeight']) + bias['ballOutputBias']

#This is for training on individual components, so the loss/optimization is done solely on this object
ballLoss = tf.nn.sigmoid_cross_entropy_with_logits(logits=ballPrediction, labels=ballLabels)
ballOptimizer = tf.train.AdamOptimizer(learning_rate=constants.learningrate).minimize(ballLoss)
correct_prediction1 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(ballPrediction)), ballLabels), tf.float32)
ballAccuracy = tf.reduce_mean(correct_prediction1)


#Bottle CNN
weights['bottleWeight'] = tf.Variable(tf.random_normal([7,7,3,constants.FILTERCOUNT])) #This is the same as the 'filter' variable (height, width, in-channels, out-channels)
bias['bottleBias'] = tf.Variable(tf.random_normal([constants.FILTERCOUNT]))
bottleConv = tf.nn.conv2d(input=bottleData, filter=weights['bottleWeight'], strides=[1,1,1,1], padding=constants.PADDING)
bottleActivation = tf.nn.relu(bottleConv + bias['bottleBias'])
weights['bottleOutputWeight'] = tf.Variable(tf.random_normal([tensorShape, 1]))
bias['bottleOutputBias'] = tf.Variable(tf.random_normal([1]))
bottleOutput = tf.reshape(bottleActivation, shape=[-1, tensorShape])
bottlePrediction = tf.matmul(ballOutput, weights['bottleOutputWeight']) + bias['bottleOutputBias']

bottleLoss = tf.nn.sigmoid_cross_entropy_with_logits(logits=bottlePrediction, labels=bottleLabels)
bottleOptimizer = tf.train.AdamOptimizer(learning_rate=constants.learningrate).minimize(bottleLoss)
correct_prediction2 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(bottlePrediction)), bottleLabels), tf.float32)
bottleAccuracy = tf.reduce_mean(correct_prediction2)


#Can CNN
weights['canWeight'] = tf.Variable(tf.random_normal([7,7,3,constants.FILTERCOUNT])) #This is the same as the 'filter' variable (height, width, in-channels, out-channels)
bias['canBias'] = tf.Variable(tf.random_normal([constants.FILTERCOUNT]))
canConv = tf.nn.conv2d(input=canData, filter=weights['canWeight'], strides=[1,1,1,1], padding=constants.PADDING)
canActivation = tf.nn.relu(canConv + bias['canBias'])
weights['canOutputWeight'] = tf.Variable(tf.random_normal([tensorShape, 1]))
bias['canOutputBias'] = tf.Variable(tf.random_normal([1]))
canOutput = tf.reshape(canActivation, shape=[-1, tensorShape])
canPrediction = tf.matmul(ballOutput, weights['canOutputWeight']) + bias['canOutputBias']

canLoss = tf.nn.sigmoid_cross_entropy_with_logits(logits=canPrediction, labels=canLabels)
tf.summary.histogram('can loss', canLoss)
canOptimizer = tf.train.AdamOptimizer(learning_rate=constants.learningrate).minimize(canLoss)
correct_prediction3 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(canPrediction)), canLabels), tf.float32)
canAccuracy = tf.reduce_mean(correct_prediction3)


#Paper CNN
weights['paperWeight'] = tf.Variable(tf.random_normal([7,7,3,constants.FILTERCOUNT])) #This is the same as the 'filter' variable (height, width, in-channels, out-channels)
bias['paperBias'] = tf.Variable(tf.random_normal([constants.FILTERCOUNT]))
paperConv = tf.nn.conv2d(input=paperData, filter=weights['paperWeight'], strides=[1,1,1,1], padding=constants.PADDING)
paperActivation = tf.nn.relu(paperConv + bias['paperBias'])
weights['paperOutputWeight'] = tf.Variable(tf.random_normal([tensorShape, 1]))
bias['paperOutputBias'] = tf.Variable(tf.random_normal([1]))
paperOutput = tf.reshape(paperActivation, shape=[-1, tensorShape])
paperPrediction = tf.matmul(paperOutput, weights['paperOutputWeight']) + bias['paperOutputBias']

paperLoss = tf.nn.sigmoid_cross_entropy_with_logits(logits=paperPrediction, labels=paperLabels)
paperOptimizer = tf.train.AdamOptimizer(learning_rate=constants.learningrate).minimize(paperLoss)
correct_prediction4 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(paperPrediction)), paperLabels), tf.float32)
paperAccuracy = tf.reduce_mean(correct_prediction4)


#Background CNN
weights['backgroundWeight'] = tf.Variable(tf.random_normal([7,7,3,constants.FILTERCOUNT]))
bias['backgroundBias'] = tf.Variable(tf.random_normal([constants.FILTERCOUNT]))
backgroundConv = tf.nn.conv2d(input=backgroundData, filter=weights['backgroundWeight'], strides=[1,1,1,1], padding=constants.PADDING)
backgroundActivation = tf.nn.relu(backgroundConv + bias['backgroundBias'])
weights['backgroundOutputWeight'] = tf.Variable(tf.random_normal([tensorShape, 1]))
bias['backgroundOutputBias'] = tf.Variable(tf.random_normal([1]))
backgroundOutput = tf.reshape(backgroundActivation, shape=[-1, tensorShape])
backgroundPrediction = tf.matmul(backgroundOutput, weights['backgroundOutputWeight']) + bias['backgroundOutputBias']

backgroundLoss = tf.nn.sigmoid_cross_entropy_with_logits(logits=backgroundPrediction, labels=backgroundLabels)
backgroundOptimizer = tf.train.AdamOptimizer(learning_rate=constants.learningrate).minimize(backgroundLoss)
correct_prediction5 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(backgroundPrediction)), backgroundLabels), tf.float32)
backgroundAccuracy = tf.reduce_mean(correct_prediction5)



#Now that all the individual CNN's are formed, the predicitons from these join to make the final NN
stackedPredictions = tf.stack([ballPrediction, bottlePrediction, canPrediction, paperPrediction, backgroundPrediction], axis=1)
rawPredictions = stackedPredictions[:, :, 0]
# print(f"Raw predictions: {np.array(rawPredictions)}")

weights['final'] = tf.Variable(tf.random_normal([5,5])) #5 classes of objects
bias['final'] = tf.Variable(tf.random_normal([5]))
finalPrediction = tf.matmul(rawPredictions, weights['final']) + bias['final']
# print(f"Final prediction: {np.array(finalPrediction)}")
finalLoss = tf.nn.sigmoid_cross_entropy_with_logits(logits=finalPrediction, labels=collectiveLabels)
# print(f"final loss {finalLoss}")
finalOptimizer = tf.train.AdamOptimizer(learning_rate=constants.learningrate).minimize(finalLoss)
correct_prediction6 = tf.cast(tf.equal(tf.argmax(tf.nn.sigmoid(finalPrediction),1), tf.argmax(collectiveLabels, 1)), tf.float32)
groupAccuracy = tf.reduce_mean(correct_prediction6)

#Need to initialize the arrays of bias/weights to be used within the session
init = tf.global_variables_initializer()
#Create saver object to save models
saver = tf.train.Saver()

#Now for the learning/training of the model, as well as the validation
if (sys.argv[1] == 'train'):
    with tf.Session() as sess:
        sess.run(init)
        merged = tf.summary.merge_all()

        trainingAcc = 0.0
        modelPath = "model"
        logdir = "log/traininglog.txt"
        if not os.path.exists(modelPath):
            os.makedirs(modelPath)
        if not os.path.exists('log'):
            os.makedirs('log')
        
        for epoch in range(constants.EPOCHCOUNT):
            batchData, batchLabels = helper_methods.getBatchInfo(constants.BATCHSIZE, 'ball')
            ballOptimizer.run(feed_dict={ballData: batchData, ballLabels: batchLabels})

            batchData, batchLabels = helper_methods.getBatchInfo(constants.BATCHSIZE, 'bottle')
            bottleOptimizer.run(feed_dict={bottleData: batchData, bottleLabels: batchLabels})

            batchData, batchLabels = helper_methods.getBatchInfo(constants.BATCHSIZE, 'can')
            canOptimizer.run(feed_dict={canData: batchData, canLabels: batchLabels})

            batchData, batchLabels = helper_methods.getBatchInfo(constants.BATCHSIZE, 'paper')
            paperOptimizer.run(feed_dict={paperData: batchData, paperLabels: batchLabels})

            batchData, batchLabels = helper_methods.getBatchInfo(constants.BATCHSIZE, 'background')
            backgroundOptimizer.run(feed_dict={backgroundData: batchData, backgroundLabels: batchLabels})

            batchData, batchLabels = helper_methods.getBatchInfo(constants.BATCHSIZE)
            finalOptimizer.run({ballData: batchData, bottleData: batchData, canData: batchData, paperData: batchData, backgroundData: batchData, collectiveLabels: batchLabels})


            if epoch % 1 == 0:
                evalData, evalLabels = helper_methods.getBatchInfo(constants.BATCHSIZE, 'ball')
                ballAcc = ballAccuracy.eval({ballData: evalData, ballLabels: evalLabels})

                evalData, evalLabels = helper_methods.getBatchInfo(constants.BATCHSIZE, 'bottle')
                bottleAcc = bottleAccuracy.eval({bottleData: evalData, bottleLabels: evalLabels})

                evalData, evalLabels = helper_methods.getBatchInfo(constants.BATCHSIZE, 'can')
                canAcc = canAccuracy.eval({canData: evalData, canLabels: evalLabels})
                
                evalData, evalLabels = helper_methods.getBatchInfo(constants.BATCHSIZE, 'paper')
                paperAcc = paperAccuracy.eval({paperData: evalData, paperLabels: evalLabels})

                evalData, evalLabels = helper_methods.getBatchInfo(constants.BATCHSIZE, 'background')
                backgroundAcc = backgroundAccuracy.eval({backgroundData: evalData, backgroundLabels: evalLabels})

                evalData, evalLabels = helper_methods.getBatchInfo(constants.BATCHSIZE)
                totalAcc = groupAccuracy.eval({ballData: evalData, bottleData: evalData, canData: evalData, paperData: evalData, backgroundData: evalData, collectiveLabels: evalLabels})

                if totalAcc >= trainingAcc:
                    trainingAcc = totalAcc
                    savePath = saver.save(sess, 'model/cnn_model.ckpt')
                    print("Highest accuracy model found, model saved")

                print("epoch: %d\tball: %.4f\tbottle: %.4f\tcan: %.4f\tpaper: %.4f\tbackground: %.4f" % (epoch, ballAcc, bottleAcc, canAcc, paperAcc, backgroundAcc))
                print("total accuracy: %.4f" % (totalAcc))

                with open(logdir, 'a') as results:
                    results.write("epoch: %d\tball: %.4f\tbottle: %.4f\tcan: %.4f\tpaper: %.4f\tbackground: %.4f\ntotal accuracy: %.4f" % (epoch, ballAcc, bottleAcc, canAcc, paperAcc, backgroundAcc, totalAcc))
    
            