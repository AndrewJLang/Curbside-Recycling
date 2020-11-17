import cv2
import pymeanshift as pms
import os

#NOTE: This will not work on my computer, needs to be run on box where pms is installed and setup

# try:
#     # Creating a folder named data
#     if not os.path.exists('updated_data'):
#         os.makedirs('updated_data')
# # If not created, then raise error
# except OSError:
#     print ('Error: Creating directory')

#I commented out
# original_image = cv2.imread("frame48.jpg")
# (segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=10, range_radius=10, min_density=300)
# status = cv2.imwrite("updated_300.png", segmented_image)
# print("Image written to file-system : ", status)


#Andrew's changes to filter through all images

def pmsTransformation():
    #file path(s) to different objects
    filePath = "../../separated_Cenek_images/"

    #Traverse all the folders that have the name of the images
    for x in os.listdir(filePath):
        #Traverse all the folders of each video
        for i in os.listdir(filePath + x):
            #Traverse the images in each of the folder
            for n in os.listdir(filePath + x + "/" + i):
                path = filePath + x +"/" + i + "/" + n
                writePath = "pms_images/" + x + "_" + n
                original_image = cv2.imread(path)
                (segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=10, range_radius=10, min_density=300)
                status = cv2.imwrite(writePath, segmented_image)


#pmsTransformation()


def testPMS():
    image = "test_images/image00005.jpg"
    image_obj = cv2.imread(image)
    (segmented_image, labels_image, number_regions) = pms.segment(image_obj, spatial_radius=10, range_radius=10, min_density=300)
    status = cv2.imwrite("test_images/test.jpg", segmented_image)
    print(type(labels_image))
    print("Labelled array shape: %s", labels_image.shape)
    print("Number of regions:\t" + str(number_regions))
    print("label image:\t" + str(labels_image))

testPMS()

cv2.destroyAllWindows()

