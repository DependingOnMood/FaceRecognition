import os
from PIL import Image
import shutil

inputRoot = "/Users/Dongbo/Documents/Education/NYU/Courses/2015 Fall/Foundations Of Machine Learning/Project/project/FaceRecognition/data"
outputRoot = "/Users/Dongbo/Documents/Education/NYU/Courses/2015 Fall/Foundations Of Machine Learning/Project/project/FaceRecognition/managed_data"

def resizePhoto(size):
    if os.path.exists(outputRoot):
        shutil.rmtree(outputRoot)
    os.mkdir(outputRoot)

    for inputFiles in os.listdir(inputRoot):
        if inputFiles !='.DS_Store':
            inputDir = inputRoot + "/" + inputFiles
            outputDir = outputRoot + "/" + inputFiles
            os.mkdir(outputDir)

            for inputImages in os.listdir(inputDir):
                 if inputImages !='.DS_Store':
                    inputImagePath = inputDir + "/" + inputImages
                    outputImagePath = outputDir + "/" + inputImages

                    try:
                        print("Resized: " + inputImagePath)
                        im = Image.open(inputImagePath)
                        im.thumbnail(size)
                        im.save(outputImagePath, "JPEG")
                    except IOError:
                        print("cannot create thumbnail for '%s'" % inputImages)


# resize_photo(size = (188 ,250))