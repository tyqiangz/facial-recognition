import numpy as np
import cv2
from matplotlib import pyplot as plt
import face_recognition
from PIL import Image, ImageDraw

from sklearn.cluster import KMeans

def calc_face_distance(image_path_1, image_path_2):
    image_1 = face_recognition.load_image_file(image_path_1)
    image_2 = face_recognition.load_image_file(image_path_2)

    # Get the face encodings for the images
    face_encoding_1 = face_recognition.face_encodings(image_1)
    face_encoding_2 = face_recognition.face_encodings(image_2)
    
    # calculate no. of faces found
    num_faces_1 = len(face_encoding_1)
    num_faces_2 = len(face_encoding_2)
    
    print(f"{num_faces_1} face(s) found in {image_path_1}")
    print(f"{num_faces_2} face(s) found in {image_path_2}")
    
    # if a single face is found in both images, then calculate similarity score
    if num_faces_1 == 1 and num_faces_2 == 1:
        face_distances = face_recognition.face_distance([face_encoding_1[0]], face_encoding_2[0])
        return 1 - face_distances
    
    else:
        return None

def plot_faces(image_path):
    '''
    Given a path to the image, extract the face(s) if any, then
    plot the face and the outline of the features.

    :param image_path: a path to the image
    :type image_path: str

    :return: None
    :rtype: None
    '''
    
    image = None

    try:
        image = face_recognition.load_image_file(image_path)
    except:
        print(f"Cannot open image at {image_path}")
    
    if image is not None:
        # generate white image same size as original image
        faces_image = np.ones_like(image) * 255

        # find all the boundary box(es) of the face(s)        
        face_locations = face_recognition.face_locations(image)

        # for all faces, copy it to the white image
        for face_location in face_locations:    
            top, right, bottom, left = face_location
            faces_image[top:bottom, left:right] = image[top:bottom, left:right]
        
        pil_image = Image.fromarray(faces_image)
        
        # show the image in a window
        pil_image.show(title="face")

        
        # Find all facial features in all the faces in the image
        face_landmarks_list = face_recognition.face_landmarks(image, model='large')
        
        # generate white image same size as original image
        pil_image = Image.fromarray(np.ones_like(image) * 255)

        # for all facial feature, e.g. eyes, nose, lips, plot the outline
        for face_landmarks in face_landmarks_list:
            d = ImageDraw.Draw(pil_image, 'RGB')
            
            for face_landmark in face_landmarks:
                d.line(face_landmarks[face_landmark], fill=(0, 0, 0))
        
        # show the image in a window
        pil_image.show(title="face outline(s)")

def extract_skin(img, face_location, face_landmarks):
    '''
    Given an image, location of the face and face landmarks,
    extract out the left and right eyes, top and bottom lips.
    
    :param img: an image read using Python Imaging Library (PIL)
    :type img: PIL.Image.Image
    
    :param face_location: coordinates indicating where the face is in `img`
    :type face_location: list of length 4, representing top, right, bottom, left
    
    :param face_landmarks: a dictionary of list of points that describe the points of each 
        face landmark, for e.g.
        ```
        {'left_eye': [(520, 132), (533, 123), (549, 122), (564, 124), (579, 130)],
         'right_eye': [(604, 132), (620, 128), (635, 128), (650, 132), (660, 142)]}
        ```
    :type face_landmarks: dictionary of iterable of iterable of 2 int
    '''
    
    # convert to numpy (for convenience)
    imArray = np.asarray(img)
    
    ##################### retain face area only ####################
    # create mask full of ones, essentially an image with zero-values
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    
    top, right, bottom, left = face_location
    
    face_bounding_box = [(left,top),
                         (right,top),
                         (right,bottom),
                         (left,bottom)]
    
    d = ImageDraw.Draw(maskIm)
    
    # extract only the face
    d.polygon(face_landmarks['chin'], outline=1, fill=1)

    # obtain the pixel values from the PIL image
    mask = np.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = np.empty(imArray.shape,dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:,:,:3] = imArray[:,:,:3]

    # transparency (4th column)
    newImArray[:,:,3] = mask*255
    
    ##################### remove face landmarks ####################
    # create mask full of ones    
    d = ImageDraw.Draw(maskIm)
    
    # for each facial landmark ...
    for face_landmark in face_landmarks:
        # do not remove the chin area
        if face_landmark == 'chin':
            continue
        
        # create a polygon full of zeros wihtin the mask of ones
        polygon = face_landmarks[face_landmark]
        d.polygon(polygon, outline=0, fill=0)

    # obtain the pixel values from the PIL image
    mask = np.array(maskIm)

    # assemble new image (uint8: 0-255)
    finalImArray = np.empty(newImArray.shape,dtype='uint8')

    # colors (three first columns, RGB)
    finalImArray[:,:,:3] = newImArray[:,:,:3]

    # transparency (4th column)
    finalImArray[:,:,3] = mask*255

    # back to Image from numpy
    newIm = Image.fromarray(finalImArray, "RGBA")
    
    return newIm

def get_skin_tone(skin_rgb, method='mean', num_tones=1):
    '''
    Extract the skin tone (RGB values) from a list of RGB values of the skin.
    
    :param skin_rgb: an array of RGB values of skin
    :type skin_rgb: numpy.ndarray
    
    :param method: the method to extract the skin tones
    :type method: str
    
    :param num_tones: number of top skin tones to extract, only applicable if 
        `method = 'knn'`
    :type num_tones: int
    
    :return skin_tone: the RGB value(s) of the skin tone of the input skin rgb
    :rtype skin_tone: numpy.ndarray
    '''
    
    if method == 'mean':
        skin_tone = np.asarray(np.mean(skin_colours, axis=0), dtype=int)
        
        return skin_tone
    
    elif method == 'median':
        skin_tone = np.asarray(np.median(skin_colours, axis=0), dtype=int)
        
        return skin_tone
    
    elif method == 'knn':
        skin_tones = KMeans(n_clusters=num_tones, random_state=42).fit(skin_rgb)
        return skin_tones
    else:
        raise ValueError('Please specify a valid method to extract skin tone.')