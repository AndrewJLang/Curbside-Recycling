import cv2
# import pymeanshift as pms
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

def pmsTransformation(objectType):
    #file path(s) to different objects
    filePath = "../blue_background_sample_images/regular_speed_frames/"
    writePath = "pms_images/" + objectType

    try:
    # Creating a folder named data
        if not os.path.exists(writePath):
            os.makedirs(writePath)
        else:
            print('folder already exists with files')
            return
    # If not created, then raise error
    except OSError:
        print (OSError)
        return

    for x in os.listdir(filePath + objectType + "/"):
        for i in os.listdir(filePath + objectType + "/" + x):
            path = filePath + objectType + "/" + x + "/" + i
            print(path)
            original_image = cv2.imread(path)
            # (segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=10, range_radius=10, min_density=300)
            status = cv2.imwrite(writePath + "/video_" + x + "_frame_" + i + ".jpg", original_image)


pmsTransformation("tennis_balls")
pmsTransformation("plastic_bottles")
pmsTransformation("soda_cans")


#NOTE: David commented out code below, should be doing same thing as code above using pms

# original_image = cv2.imread("data/frame48")
# (segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6, range_radius=4.5, min_density=50)
# status = cv2.imwrite("updated48.png", segmented_image)
# print("Image written to file-system : ", status)


# counter = 0
# for filename in os.listdir("data/"):
    # print(filename)
    # original_image = cv2.imread("data/" + filename)
    # (segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6, range_radius=4.5, min_density=50)
    # status = cv2.imwrite("./updated_data/updated_frame" + str(counter) + ".jpg", segmented_image)
    # print("Image written to file-system: :", status)


cv2.destroyAllWindows()

# original_image = cv2.imread("example.png")
# (segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=6, range_radius=4.5, min_density=50)
# status = cv2.imwrite("updated_ex.png", segmented_image)
# print("Image written to file-system : ", status)