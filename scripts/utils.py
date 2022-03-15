import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

import face_recognition
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def distance(embeddings1, embeddings2, distance_metric='euclidean'):
    '''
    Distance metric for 2 embedding vectors.
    
    :param embeddings1: first embedding, shape of 1 x N
    :type embeddings1: numpy.ndarray

    :param embeddings2: second embedding, shape of 1 x N
    :type embeddings2: numpy.ndarray
    
    :param distance_metric: the distance metric to use to compare similarity
        between two embedding vectors.
    :type distance_metric: str
    
    :return dist: distance between the embedding vectors based on the selected distance metric
    :rtype dist: float
    '''

    if distance_metric=='euclidean':
        # Euclidian distance
        dist = np.linalg.norm(embeddings1 - embeddings2)
    
    elif distance_metric=='cosine':
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2))
        norm = np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
        similarity = dot / norm
        
        # to round down for round-off errors where `similarity = 1.00001`
        if similarity > 1:
            similarity = 1
        if similarity < -1:
            similarity = -1
        
        dist = np.arccos(similarity) / math.pi
    else:
        raise f"Undefined distance metric: {distance_metric}"

    return dist

def get_face_similarity(image_path_1, image_path_2, method="facenet"):
    '''
    Compute a similarity score of the faces in 2 images.
    1 being exactly the same face, 0 being totally different.

    `None` will be returned if either of the images do not 
    contain exactly one face.

    :param image_path_1: filepath to 1st image
    :type image_path_1: str

    :param image_path_2: filepath to 2nd image
    :type image_path_2: str

    :param method: the method used to detect faces, 
        `method = "dlib"` is faster but less accurate,
        `method = "facenet"` is more accurate but slower .
    :type method: str

    :return face_similarity: a similarity score of the faces in the images given.
    :rtype face_similarity: `float` if exactly 1 face is detected in each image, else `None`.
    '''
    if method == "dlib":
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
            # convert the numpy array to float
            face_similarity = 1 - float(face_distances)
            return face_similarity
        # else return None
        else:
            return None

    elif method == "facenet":
        embeddings = []
        IMAGE_PATHS = [image_path_1, image_path_2]

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # for detecting bounding box for face(s)
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            keep_all=True, device=device
        )

        # for generating embeddings for a face image
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        for i in range(len(IMAGE_PATHS)):
            image_path = IMAGE_PATHS[i]
            
            image = Image.open(image_path).convert("RGB")
            
            face_images, prob = mtcnn(image, return_prob=True)
            
            if face_images is None:
                print(f"0 face(s) found in {image_path}")
                embeddings.append(None)

            else:
                print(f"{len(face_images)} face(s) found in {image_path}")

                if len(face_images) > 1:
                    embeddings.append(None)
                elif len(face_images) == 1:
                    embedding = resnet(face_images).detach().cpu()
                    embeddings.append(embedding[0])
                
        e1 = embeddings[0]
        e2 = embeddings[1]

        if e1 is None or e2 is None:
            return None

        face_similarity = 1 - distance(e1.numpy(), e2.numpy(), distance_metric='cosine')
        return face_similarity

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

def extract_skin_colours(img, face_landmarks):
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
    
    :return skin_colours: rgb values of the pixels identified as the skin
    :rtype skin_colours: numpy.ndarray
    
    :return skin_img: an image showing only the skin, the rest of the non-skin areas of the 
        image are in white.
    :rtype skin_img: PIL.Image.Image
    '''
    
    # convert to numpy (for convenience)
    imArray = np.asarray(img)
    
    ##################### retain face area only ####################
    # create mask full of ones, essentially an image with zero-values
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    
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
    skin_img = Image.fromarray(finalImArray, "RGBA")
    
    ################### extract skin colours ###################
    skin_colours = []
    pixel_values = np.asarray(skin_img)

    height, width, channels = pixel_values.shape

    for i in range(height):
        for j in range(width):
            colour = pixel_values[i][j]

            if colour[3] != 0:
                skin_colours.append(colour[:3])

    skin_colours = np.asarray(skin_colours)
    
    return skin_colours, skin_img

def get_skin_tone(skin_rgb, method='median', num_tones=2):
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
    
    Example:
    ```
    >>> skin_rgb = skin_rgb = np.array([[148, 128, 121],
                                        [144, 124, 117],
                                        [142, 121, 116],
                                        [106,  75,  73],
                                        [106,  75,  73],
                                        [111,  81,  83]])
    >>> get_skin_tone(skin_rgb, method='knn', num_tones=2)
    
    [[107,  77,  76],
     [144, 124, 118]]
     
    >>> skin_rgb = skin_rgb = np.array([[148, 128, 121],
                                        [144, 124, 117],
                                        [142, 121, 116],
                                        [106,  75,  73],
                                        [106,  75,  73],
                                        [111,  81,  83]])
    >>> get_skin_tone(skin_rgb, method='median', num_tones=2)
    
    [[126, 101,  99]]
    ```
    '''
    
    # mean of the rgb values
    if method == 'mean':
        skin_tone = np.asarray(np.mean(skin_rgb, axis=0), dtype=int)
        
        # make it an array of array for consistency of return type
        skin_tones = np.array([skin_tone])
        return skin_tones
    
    # median of the rgb values
    elif method == 'median':
        skin_tone = np.asarray(np.median(skin_rgb, axis=0), dtype=int)
        # make it an array of array for consistency of return type
        skin_tones = np.array([skin_tone])
        return skin_tones
    
    # K nearest neighbours of the rgb values
    elif method == 'knn':
        skin_clusters = KMeans(n_clusters=num_tones, random_state=42).fit(skin_rgb)
        skin_tones = np.array(skin_clusters.cluster_centers_, dtype=int)
        return skin_tones
    else:
        raise ValueError('Please specify a valid method to extract skin tone.')

def get_skin_tone_similarity(rgb1, rgb2, method="mean"):
    '''
    Obtain the percentage similarity between 2 skin tones.
    A score between 0 to 1 is returned, 0 meaning totally dissimilar,
    1 meaning exactly the same.
    
    A research paper about the best distance metric of two skin rgb values can be found in: 
    http://dx.doi.org/10.5121/csit.2013.3210.
    
    :param rgb1: rgb value(s) of skin tone 1
    :type rgb1: numpy.ndarray
    
    :param rgb2: rgb value(s) of skin tone 2
    :type rgb2: numpy.ndarray
    
    :return skin_tone_similarity: percentage similarity between 2 skin tones
    :rtype skin_tone_similarity: float
    '''
    
    assert rgb1.shape == rgb2.shape
    num_skin_tones = len(rgb1)
    
    skin_tone_similarities = 0
    
    for i in range(num_skin_tones):
        abs_diff = np.abs(rgb1[i] - rgb2[i])
        skin_tone_similarities += 1 - np.mean(abs_diff / 255)
    
    skin_tone_similarity = skin_tone_similarities / num_skin_tones
    
    return skin_tone_similarity

def get_face_landmarks(image_path, method="dlib"):
    '''
    Return the facial landmarks if one and only one face is found in `image_path`.
    
    :param image_path: filepath to an image
    :type image_path: str
    
    :param method: method used to detect facial landmarks
    :type method: str
    
    :return face_landmarks: a dictionary of face landmark and the coordinates
    :rtype face_landmarks: dict
    '''
    
    if method == "dlib":
        # load the image file
        image = face_recognition.load_image_file(image_path)
        
        # detect the bounding box containing the face
        face_locations = face_recognition.face_locations(image)
        
        # if more than 1 face is detected, `None` value is returned
        if len(face_locations) != 1:
            return None, None
        
        # obtain coordinates of the facial features, `chin`, `eyes` etc
        face_landmarks_list = face_recognition.face_landmarks(image, face_locations, model='large')
        
        # if no facial features are detected or more than 1 set of facial features are detected, 
        # `None` values are returned
        if len(face_landmarks_list) != 1:
            return None, None
    
        face_landmarks = face_landmarks_list[0]
        
        return face_landmarks, image
    
    elif method == "facenet":
        image = Image.open(image_path).convert("RGB")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mtcnn = MTCNN(
            image_size=160, margin=20, min_face_size=20, keep_all=True,
            device=device
        )

        tensor_image = mtcnn(image)
        
        # if more than 1 face is detected or no face is detected, `None` values are returned
        if tensor_image is None or tensor_image.shape[0] != 1:
            return None, None
        
        # convert the tensor image to RGB valued image in numpy 
        image = np.array(((tensor_image[0]+1)/2 * 255).permute(1, 2, 0), dtype='uint8')
        
        # obtain coordinates of the facial features, `chin`, `eyes` etc
        face_landmarks_list = face_recognition.face_landmarks(image, model='large')
        
        # if no facial features are detected or more than 1 set of facial features are detected, 
        # `None` values are returned
        if len(face_landmarks_list) != 1:
            return None, None
        
        face_landmarks = face_landmarks_list[0]
        
        return face_landmarks, image
        
def get_skin_tone_similarity_face(image_path_1, image_path_2):
    '''
    Obtain the percentage similarity between 2 skin tones from 2 images.
    
    If exactly 1 face is found in each image, a score between 0 to 1 is returned, 
    0 meaning totally dissimilar, 1 meaning exactly the same;
    else, return None
    
    :param image_path_1: filepath to the first image
    :type image_path_1: str
    
    :param image_path_2: filepath to the second image
    :type image_path_2: str
    
    :return skin_tone_similarity: percentage similarity between 2 skin tones
    :rtype skin_tone_similarity: float
    
    :return skin_1: an image showing only the skin of `image_path_1`, 
        the rest of the non-skin areas of the image are in white.
    :rtype skin_1: PIL.Image.Image
    
    :return skin_2: an image showing only the skin of `image_path_2`, 
        the rest of the non-skin areas of the image are in white.
    :rtype skin_2: PIL.Image.Image
    
    :return skin_tone_1: rgb value of skin tone of `image_path_1`
    :rtype skin_tone_1: numpy.ndarray
    
    :return skin_tone_2: rgb value of skin tone of `image_path_2`
    :rtype skin_tone_2: numpy.ndarray
    '''
    
    skin_tones = []
    skin_imgs = []
    
    for image_path in [image_path_1, image_path_2]:        
        # try obtaining facial landmarks using facenet
        face_landmarks, image = get_face_landmarks(image_path, method="facenet")
        
        if face_landmarks is not None:
            image = Image.fromarray(image).convert("RGBA") 
        
        #  if facenet fails, try obtaining facial landmarks using dlib
        else:
            face_landmarks, image = get_face_landmarks(image_path, method="dlib")
            
            # if both method fails, return none
            if face_landmarks is None:
                return None, None, None, None, None
            else:
                image = Image.open(image_path).convert("RGBA")
        
        # extract the RGB pixel values of the skin colour on the face
        skin_colours, skin_img = extract_skin_colours(image, face_landmarks)
        
        # compute a skin tone value from the collection of skin RGB colours
        skin_tone = get_skin_tone(skin_colours, method='knn')
        
        skin_imgs.append(skin_img)
        skin_tones.append(skin_tone)
    
    # compute a similarity score between 2 collection of skin tones
    skin_tone_similarity = get_skin_tone_similarity(skin_tones[0], skin_tones[1])
    
    skin_1 = skin_imgs[0]
    skin_2 = skin_imgs[1]
    skin_tone_1 = skin_tones[0]
    skin_tone_2 = skin_tones[1]
    
    return skin_tone_similarity, skin_1, skin_2, skin_tone_1, skin_tone_2

def rgb_to_hex(rgb):
    '''
    Converts RGB values to hex values.
    
    :param rgb: rgb value
    :type rgb: iterable of length 3
    
    :return hex_code: a hex code for `rgb`
    :rtype hex_code: str
    
    Example:
    ```
    >>> rgb_to_hex([255,255,255])
    '#ffffff'
    ```
    '''
    int_to_hex = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4',
                  5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
                  10:'A', 11:'B', 12:'C', 13:'D',
                  14:'E', 15:'F'}
    hex_code = '#'
    
    for channel in rgb:
        hex_code += hex(channel)[-2:]
    
    return hex_code