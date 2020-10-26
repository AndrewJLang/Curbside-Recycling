

KEEP_RATE = 0.8                     #Rate of dropping out in the dropout layer
LOG_DIR = "../ops_logs"             #Directory where the logs would be stored for visualization of the training

#Neural network constants (change these)
cat1_dir = "pymeanshift/pms_images/paper/"
cat2_dir = "pymeanshift/pms_images/plastic_bottles/"
cat3_dir = "pymeanshift/pms_images/cardboard/"
cat4_dir = "pymeanshift/pms_images/metal_cans/"
cat5_dir = "pymeanshift/pms_images/plastic_bags/"
cat6_dir = "pymeanshift/pms_images/shells/"
CAT1            = "paper"
CAT2            = "plastic_bottles"
CAT3            = "cardboard"
CAT4            = "metal_cans"
CAT5            = "plastic_bags"
CAT6            = "shells"
CAT1_ONEHOT     = [1,0,0,0,0,0]
CAT2_ONEHOT     = [0,1,0,0,0,0]
CAT3_ONEHOT     = [0,0,1,0,0,0]
CAT4_ONEHOT     = [0,0,0,1,0,0]
CAT5_ONEHOT     = [0,0,0,0,1,0]
CAT6_ONEHOT     = [0,0,0,0,0,1]
LEARNING_RATE = 0.01               #Learning rate for training the CNN
CNN_LOCAL1 = 32                  #Number of features output for conv layer 1
CNN_LOCAL2 = 32                  #Number of features output for conv layer 1
CNN_GLOBAL1 = 64                  #Number of features output for conv layer 2
CNN_CLASSES      = 6
CNN_EPOCHS       = 1000
CNN_FULL1   = 400                #Number of features output for fully connected layer1
IMG_SIZE = 56
FULL_IMGSIZE = 1000
IMG_DEPTH   = 3
KEEP_RATE = 0.85
BATCH_SIZE = 200
