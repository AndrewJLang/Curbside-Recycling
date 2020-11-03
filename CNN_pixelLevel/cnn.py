#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for TBI, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2
import math
import sys
import os

#Python Modules
import constants
import featureReader

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

###################################################################

###################################################################
#1. Convolutional layer
#2. Pooling layers
#3. Convolutional layer
#4. pooling layer
#5. Fully connected layer
#6. Logits layer
###################################################################

####################################################################################################################################
#Helper Functions
####################################################################################################################################
def outputResults(image,mask,fout='segmentation.png'):
    #create the segmented image
    canvas = image.copy()
    canvas[mask == -1] = [0,0,0]
    canvas[mask == 0] = [0,0,255]
    canvas[mask == 1] = [0,255,0]
    canvas[mask == 2] = [255,0,0]
    canvas[mask == 3] = [0,255,255]
    canvas[mask == 4] = [255,0,255]
    canvas[mask == 5] = [255,255,0]

    #show the original image and the segmented image and then save the results
    cv2.imwrite(fout,canvas)

    #count the percentage of each category
    cat1_count = np.count_nonzero(mask == 0)
    cat2_count = np.count_nonzero(mask == 1)
    cat3_count = np.count_nonzero(mask == 2)
    cat4_count = np.count_nonzero(mask == 3)
    cat5_count = np.count_nonzero(mask == 4)
    cat6_count = np.count_nonzero(mask == 5)
    total = cat1_count + cat2_count + cat3_count + cat4_count + cat5_count + cat6_count

    #get the percentage of each category
    p1 = cat1_count / total
    p2 = cat2_count / total
    p3 = cat3_count / total
    p4 = cat4_count / total
    p5 = cat5_count / total
    p6 = cat6_count / total

    #output to text file
    with open('results.txt','a') as f:
        f.write("\nusing model: %s\n" % sys.argv[3])
        f.write("evaluate image: %s\n\n" % sys.argv[2])
        f.write("--------------------------------------------------------------------------------------\n")
        f.write("%s : %f\n" % (constants.CAT1,p1))
        f.write("%s : %f\n" % (constants.CAT2,p2))
        f.write("%s : %f\n" % (constants.CAT3,p3))
        f.write("%s : %f\n" % (constants.CAT4,p4))
        f.write("%s : %f\n" % (constants.CAT5,p5))
        f.write("%s : %f\n" % (constants.CAT6,p6))
        f.write("--------------------------------------------------------------------------------------\n")
        f.write("------------------------------------END-----------------------------------------------\n")
        f.write("--------------------------------------------------------------------------------------\n")

        greatest = max(cat1_count,cat2_count,cat3_count,cat4_count)

        #f.write out to the terminal what the most common category was for the image
        if(greatest == cat1_count):
            f.write("\nthe most common category is: " + constants.CAT1)
        elif(greatest == cat2_count):
            f.write("\nthe most common category is: " + constants.CAT2)
        elif(greatest == cat3_count):
            f.write("\nthe most common category is: " + constants.CAT3)
        elif(greatest == cat4_count):
            f.write("\nthe most common category is: " + constants.CAT4)
        elif(greatest == cat5_count):
            f.write("\nthe most common category is: " + constants.CAT5)
        elif(greatest == cat6_count):
            f.write("\nthe most common category is: " + constants.CAT6)
        else:
            f.write("\nsorry something went wrong counting the predictions")


####################################################################################################################################
#######################################################################################
#######################################################################################
#Main Function

def main(unused_argv):

    #check the number of arguments given with running the program
    #must be at least two
    #argv[1] is the mode of operation {test,see,train}
    #argv[2] is the input image
    #argv[3] is the optional
    if not os.path.exists('log'):
        os.makedirs('log')

    if len(sys.argv) >= 2:

        #################################################################################################################
        #################################################################################################################
        #Define our Convolutionary Neural Network from scratch
        x = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x1 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x2 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x3 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x4 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x5 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x6 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        y = tf.placeholder('float',[None,constants.CLASSES])
        y1 = tf.placeholder('float',[None,1])
        y2 = tf.placeholder('float',[None,1])
        y3 = tf.placeholder('float',[None,1])
        y4 = tf.placeholder('float',[None,1])
        y5 = tf.placeholder('float',[None,1])
        y6 = tf.placeholder('float',[None,1])

        weights = {}
        biases = {}

        #magic number = width * height * n_convout
        magic_number = int(constants.IMG_SIZE * constants.IMG_SIZE * constants.CNN_LOCAL1)


        #tree matter convolution
        weights['w_treematter'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
        biases['b_treematter'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
        conv_treematter = tf.nn.conv2d(x1,weights['w_treematter'],strides=[1,1,1,1],padding='SAME',name='local1')
        activation1 = tf.nn.relu(conv_treematter + biases['b_treematter'])
        weights['out1'] = tf.Variable(tf.random_normal([magic_number,1]))
        biases['out1'] = tf.Variable(tf.random_normal([1]))
        output1 = tf.reshape(activation1,[-1,magic_number])
        predictions1 = tf.matmul(output1,weights['out1'])+biases['out1']

        #plwood convolution
        weights['w_plwood'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
        biases['b_plwood'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
        conv_plwood = tf.nn.conv2d(x2,weights['w_plwood'],strides=[1,1,1,1],padding='SAME',name='local1')
        activation2 = tf.nn.relu(conv_plwood + biases['b_plwood'])
        weights['out2'] = tf.Variable(tf.random_normal([magic_number,1]))
        biases['out2'] = tf.Variable(tf.random_normal([1]))
        output2 = tf.reshape(activation2,[-1,magic_number])
        predictions2 = tf.matmul(output2,weights['out2'])+biases['out2']

        #cardboard convolution
        weights['w_cardboard'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
        biases['b_cardboard'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
        conv_cardboard = tf.nn.conv2d(x3,weights['w_cardboard'],strides=[1,1,1,1],padding='SAME',name='local1')
        activation3 = tf.nn.relu(conv_cardboard + biases['b_cardboard'])
        weights['out3'] = tf.Variable(tf.random_normal([magic_number,1]))
        biases['out3'] = tf.Variable(tf.random_normal([1]))
        output3 = tf.reshape(activation3,[-1,magic_number])
        predictions3 = tf.matmul(output3,weights['out3'])+biases['out3']

        #bottles convolution
        weights['w_bottles'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
        biases['b_bottles'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
        conv_bottles = tf.nn.conv2d(x4,weights['w_bottles'],strides=[1,1,1,1],padding='SAME',name='local1')
        activation4 = tf.nn.relu(conv_bottles + biases['b_bottles'])
        weights['out4'] = tf.Variable(tf.random_normal([magic_number,1]))
        biases['out4'] = tf.Variable(tf.random_normal([1]))
        output4 = tf.reshape(activation4,[-1,magic_number])
        predictions4 = tf.matmul(output4,weights['out4'])+biases['out4']

        #trashbag convolution
        weights['w_trashbag'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
        biases['b_trashbag'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
        conv_trashbag = tf.nn.conv2d(x5,weights['w_trashbag'],strides=[1,1,1,1],padding='SAME',name='local1')
        activation5 = tf.nn.relu(conv_trashbag + biases['b_trashbag'])
        weights['out5'] = tf.Variable(tf.random_normal([magic_number,1]))
        biases['out5'] = tf.Variable(tf.random_normal([1]))
        output5 = tf.reshape(activation5,[-1,magic_number])
        predictions5 = tf.matmul(output5,weights['out5'])+biases['out5']

        #blackbag convolution
        weights['w_blackbag'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
        biases['b_blackbag'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
        conv_blackbag = tf.nn.conv2d(x6,weights['w_blackbag'],strides=[1,1,1,1],padding='SAME',name='local1')
        activation6 = tf.nn.relu(conv_blackbag + biases['b_blackbag'])
        weights['out6'] = tf.Variable(tf.random_normal([magic_number,1]))
        biases['out6'] = tf.Variable(tf.random_normal([1]))
        output6 = tf.reshape(activation6,[-1,magic_number])
        predictions6 = tf.matmul(output6,weights['out6'])+biases['out6']

        #define optimization and accuracy creation
        with tf.name_scope('cost'):
            cost1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions1,labels=y1)
            cost2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions2,labels=y2)
            cost3 = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions3,labels=y3)
            cost4 = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions4,labels=y4)
            cost5 = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions5,labels=y5)
            cost6 = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions6,labels=y6)
            tf.summary.histogram('cost1',cost1)
            tf.summary.histogram('cost2',cost2)
            tf.summary.histogram('cost3',cost3)
            tf.summary.histogram('cost4',cost4)
            tf.summary.histogram('cost5',cost5)
            tf.summary.histogram('cost6',cost6)
        with tf.name_scope('optimizer'):
            optimizer1= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost1)
            optimizer2= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost2)
            optimizer3= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost3)
            optimizer4= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost4)
            optimizer5= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost5)
            optimizer6= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost6)
        with tf.name_scope('accuracy'):
            correct_prediction1 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions1)),y1),tf.float32)
            correct_prediction2 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions2)),y2),tf.float32)
            correct_prediction3 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions3)),y3),tf.float32)
            correct_prediction4 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions4)),y4),tf.float32)
            correct_prediction5 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions5)),y5),tf.float32)
            correct_prediction6 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions6)),y6),tf.float32)

            accuracy1 = tf.reduce_mean(correct_prediction1)
            accuracy2 = tf.reduce_mean(correct_prediction2)
            accuracy3 = tf.reduce_mean(correct_prediction3)
            accuracy4 = tf.reduce_mean(correct_prediction4)
            accuracy5 = tf.reduce_mean(correct_prediction5)
            accuracy6 = tf.reduce_mean(correct_prediction6)

        #for testing purposes
        stacked = tf.stack([predictions1,predictions2,predictions3,predictions4,predictions5,predictions6],axis=1)
        all_raws = stacked[:,:,0]

        #final fc layer
        weights['out'] = tf.Variable(tf.random_normal([6,6]))
        biases['out'] = tf.Variable(tf.random_normal([6]))
        predictions_final = tf.matmul(all_raws,weights['out']) + biases['out']
        all_cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions_final,labels=y)
        all_optimizer = tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(all_cost)
        all_correct = tf.cast(tf.equal(tf.argmax(tf.nn.sigmoid(predictions_final),1),tf.argmax(y,1)),tf.float32)
        all_accuracy = tf.reduce_mean(all_correct)

        #################################################################################################################
        #################################################################################################################
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#helper functions

        #training mode trained on the image
        if(sys.argv[1] == 'train'):
            #Run the session/CNN and train/record accuracies at given steps
            #net = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
            #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
            with tf.Session() as sess:
                sess.run(init)
                merged = tf.summary.merge_all()

                #train the model
                acc = 0.00;
                modelpath = "model"
                logdir = 'log/traininglog.txt'
                if not os.path.exists(modelpath):
                    os.makedirs(modelpath)
                if not os.path.exists('log'):
                    os.makedirs('log')

                for epoch in range(constants.CNN_EPOCHS):

                    #get an image batch and train each model separately
                    batch_x,batch_y = featureReader.getBatch(constants.BATCH_SIZE,'treematter')
                    optimizer1.run(feed_dict={x1: batch_x, y1: batch_y})
                    batch_x,batch_y = featureReader.getBatch(constants.BATCH_SIZE,'plywood')
                    optimizer2.run(feed_dict={x2: batch_x, y2: batch_y})
                    batch_x,batch_y = featureReader.getBatch(constants.BATCH_SIZE,'cardboard')
                    optimizer3.run(feed_dict={x3: batch_x, y3: batch_y})
                    batch_x,batch_y = featureReader.getBatch(constants.BATCH_SIZE,'bottles')
                    optimizer4.run(feed_dict={x4: batch_x, y4: batch_y})
                    batch_x,batch_y = featureReader.getBatch(constants.BATCH_SIZE,'trashbag')
                    optimizer5.run(feed_dict={x5: batch_x, y5: batch_y})
                    batch_x,batch_y = featureReader.getBatch(constants.BATCH_SIZE,'blackbag')
                    optimizer6.run(feed_dict={x6: batch_x, y6: batch_y})
                    batch_x,batch_y = featureReader.getBatch(constants.BATCH_SIZE)
                    all_optimizer.run({x1: batch_x,x2: batch_x,x3: batch_x,x4: batch_x,x5: batch_x,x6: batch_x, y: batch_y})

                    #evaluate the models separately using a test set
                    if epoch % 1 == 0:
                        eval_x,eval_y = featureReader.getBatch(constants.BATCH_SIZE,'treematter')
                        acc1 = accuracy1.eval({x1: eval_x, y1: eval_y})
                        eval_x,eval_y = featureReader.getBatch(constants.BATCH_SIZE,'plywood')
                        acc2 = accuracy2.eval({x2: eval_x, y2: eval_y})
                        eval_x,eval_y = featureReader.getBatch(constants.BATCH_SIZE,'cardboard')
                        acc3 = accuracy3.eval({x3: eval_x, y3: eval_y})
                        eval_x,eval_y = featureReader.getBatch(constants.BATCH_SIZE,'bottles')
                        acc4 = accuracy4.eval({x4: eval_x, y4: eval_y})
                        eval_x,eval_y = featureReader.getBatch(constants.BATCH_SIZE,'trashbag')
                        acc5 = accuracy5.eval({x5: eval_x, y5: eval_y})
                        eval_x,eval_y = featureReader.getBatch(constants.BATCH_SIZE,'blackbag')
                        acc6 = accuracy6.eval({x6: eval_x, y6: eval_y})

                        eval_x,eval_y = featureReader.getBatch(constants.BATCH_SIZE)
                        accnew = all_accuracy.eval({x1: eval_x,x2: eval_x,x3: eval_x,x4: eval_x,x5: eval_x,x6: eval_x, y: eval_y})

                        #save the model if it holds the highest accuracy or is tied for highest accuracy
                        if(accnew >= acc):
                            acc = accnew
                            save_path = saver.save(sess,'model/cnn_model.ckpt')
                            print("highest accuracy found! model saved")

                        print('epoch: %i  treematter: %.4f  plywood: %.4f  cardboard: %.4f   bottles: %.4f  trashbag: %.4f  blackbag: %.4f  all: %.4f' % (epoch,acc1,acc2,acc3,acc4,acc5,acc6,accnew))
                        with open(logdir,'a') as log_out:
                            log_out.write('epoch: %i   treematter: %.4f  plywood: %.4f  cardboard: %.4f  bottles: %.4f  trashbag: %.4f    blackbag: %.4f  all: %.4f\n' % (epoch,acc1,acc2,acc3,acc4,acc5,acc6,accnew))


        #testing method needs a saved check point directory (model)
        elif(sys.argv[1] == 'test' and len(sys.argv) == 4):
            #get the directory of the checkpoint
            ckpt_dir = sys.argv[3]

            #read the image
            if os.path.isfile(sys.argv[2]):
                tmp = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
                h,w = tmp.shape[:2]
                if(h >= constants.FULL_IMGSIZE or w >= constants.FULL_IMGSIZE):
                    image = cv2.resize(tmp,(constants.FULL_IMGSIZE,constants.FULL_IMGSIZE),interpolation=cv2.INTER_CUBIC)
                else:
                    image = tmp

            #restore the graph and make the predictions and show the segmented image
            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess,ckpt_dir)
                print("session restored!")

                #we recreate the image by painting the best_guess mask on a blank canvas with the same shape as image
                #initialize counters and the height and width of the image being tested.
                #constants.IMG_SIZE is the img size the learned model uses for classifiying a pixel.
                #NOT THE actual size of the image being tested
                h,w = image.shape[:2]
                count = 0
                count2 = 0
                best_guess = np.full((h,w),-1)
                raw_guess = np.full((h,w,6),0)
                tmp = []
                i0 = int(constants.IMG_SIZE / 2)
                j0 = int(constants.IMG_SIZE / 2)

                #define our log file and pixel segmentation file name
                if not os.path.exists('results'):
                    os.mkdir('results')

                imgname = os.path.basename(sys.argv[2])
                modelname = os.path.dirname(sys.argv[3])
                logname = "results/rawoutput_" + str(os.path.splitext(os.path.basename(sys.argv[2]))[0]) + '_' + modelname + ".txt"
                seg_file = 'results/' + os.path.splitext(imgname)[0] + '_' + modelname + '_learnedseg' + ".png"

                #GO THROUGH EACH PIXEL WITHOUT THE EDGES SINCE WE NEED TO MAKE SURE EVERY PART OF THE PIXEL AREA
                #BEING SENT TO THE MODEL IS PART OF THE IMAGE
                for i in range(int(constants.IMG_SIZE / 2),int(len(image) - (constants.IMG_SIZE / 2))):
                    for j in range(int(constants.IMG_SIZE / 2),int(len(image[0]) - (constants.IMG_SIZE / 2))):

                        #get the bounding box around the pixel to send to the training
                        box = image[i-int(constants.IMG_SIZE / 2):i+int(constants.IMG_SIZE / 2),j-int(constants.IMG_SIZE / 2):j+int(constants.IMG_SIZE / 2)]

                        #append the box to a temporary array
                        tmp.append(box)

                        #once the temporary array is the same size as the batch size, run the testing on the batch
                        if(len(tmp) == constants.BATCH_SIZE or count == ((h - constants.IMG_SIZE) * (w - constants.IMG_SIZE)) - 1):
                            batch = np.array(tmp)
                            rawpredictions = all_raws.eval({x1:batch, x2:batch, x3:batch, x4:batch, x5:batch, x6:batch})
                            mask = rawpredictions.argmax(axis=1)

                            #now we go through the mask and insert the values to the correct position of best_guess which is a copy of
                            #the original image except all the values are -1
                            for raw,cat in zip(rawpredictions,mask):
                                best_guess[i0,j0] = cat
                                raw_guess[i0,j0] = raw
                                if j0 == (w - int(constants.IMG_SIZE/2)) - 1:
                                    j0 = int(constants.IMG_SIZE / 2)
                                    i0 += 1
                                else:
                                    j0 += 1

                            #give console output to show progress
                            outputResults(image,np.array(best_guess),fout=seg_file)
                            print('%i out of %i complete' % (count2,math.ceil(int((h - constants.IMG_SIZE) * (w - constants.IMG_SIZE) / constants.BATCH_SIZE))))
                            #empty tmporary array
                            tmp = []
                            count2 += 1
                        count += 1

                np.save(logname,raw_guess)
        else:
            print("train ")
            print("trainseg ")
            print("test [image_filepath] [model_filepath]")
    else:
        print("oopsies")
        print("argv[1]: mode of operation (test,train)")

if __name__ == "__main__":
    tf.app.run()
