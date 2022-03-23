import os
from tqdm import tqdm
import hashlib
import pickle
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

def get_face_image(image_filepath, convert_numpy=True):
    '''
    Obtain an image such that the image contains only 1 face.
    If no face(s) can be found, return `None`.
    
    :param image_filepath: filepath to the image
    :type image_filepath: str
    
    :param convert_numpy: to convert the face images to numpy data type or not
    :type convert_numpy: bool
    
    :return face_image: 
        1. if convert_numpy=False:
            a tensor object representing the face, should have dimensions
            `[num_of_channels, height, width]`, with float values ranging from -1 to 1.
        2. if convert_numpy=True:
            a numpy.ndaarray representing the face, should have dimensions
            `[height, width, num_of_channels]`, with int values ranging from 0 to 255.
    :rtype face_image: torch.Tensor or numpy.ndarray
    '''
    image = Image.open(image_filepath).convert("RGB")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # for detecting face(s), only at most 1 face will be returned, 
    # the area of the image that is most probably a face is returned
    mtcnn = MTCNN(
        image_size=200, margin=0, min_face_size=50,
        select_largest=False, keep_all=False, device=device
    )

    face_tensor_image = mtcnn(image)

    # if no faces are detected, `None` value is returned
    if face_tensor_image is None:
        return None
    
    if convert_numpy:
        # convert the tensor image to RGB valued image in numpy 
        face_image = np.array(((face_tensor_image+1)/2 * 255).permute(1, 2, 0), dtype='uint8')
    else:
        face_image = face_tensor_image
        
    return face_image

def get_face_landmarks(image_filepath):
    '''
    Return the facial landmarks if at least one face is found in `image_path`.
    
    :param image_filepath: filepath to an image
    :type image_filepath: str
    
    :return face_landmarks: a dictionary of face landmark and the coordinates
    :rtype face_landmarks: dict
    
    :return face_image: an image that corresponds to the face in which landmarks are extracted
    :rtype face_image: numpy.ndarray
    '''
    
    face_image = get_face_image(image_filepath, convert_numpy=True)
    
    # if more than 1 face is detected or no face is detected, `None` values are returned
    if face_image is None:
        return None, None

    # obtain coordinates of the facial features, `chin`, `eyes` etc
    face_landmarks_list = face_recognition.face_landmarks(face_image, model='large')

    # if no facial features are detected or more than 1 set of facial features are detected, 
    # `None` values are returned
    if len(face_landmarks_list) != 1:
        return None, None

    face_landmarks = face_landmarks_list[0]

    return face_landmarks, face_image

def get_face_embedding(image_filepath):
    '''
    Compute an face embedding for an image containing face.
    If one or more faces are found in the image, the area of the image 
    that has the highest probability of being a face is returned.

    :param image_filepath: filepath to image
    :type image_filepath: str

    :return embedding: a vector representing the face
    :rtype embedding: numpy.ndarray
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # for generating embeddings for a face image
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # detect faces
    face_image = get_face_image(image_filepath, convert_numpy=False)

    # if no faces are detected
    if face_image is None:
        embedding = None
    else:
        # add one more dimension to the tensors as the resnet takes in a tensor of dimension 4
        # `(num_img, num_channels, width, height)`
        face_images = torch.unsqueeze(face_image, dim=0)

        # obtain face embedding
        embedding = resnet(face_images).detach().cpu()[0].numpy()

    return embedding

def get_skin_colours(image_filepath):
    '''
    Given an image, location of the face and face landmarks,
    extract out the left and right eyes, top and bottom lips.
    
    :param image_filepath: filepath to image
    :type image_filepath: str
    
    :return skin_colours: rgb values of the pixels identified as the skin
    :rtype skin_colours: numpy.ndarray
    
    :return skin_img: an image showing only the skin, the rest of the non-skin areas of the 
        image are in white.
    :rtype skin_img: PIL.Image.Image
    '''
    
    # try obtaining facial landmarks using facenet
    face_landmarks, face_image = get_face_landmarks(image_filepath)

    if face_landmarks is None:
        return None, None
    
    else:
        image = Image.fromarray(face_image).convert("RGBA")
        
        # convert to numpy (for convenience)
        imArray = np.asarray(image)

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
    
def _get_skin_tones(skin_rgb, method='median', num_tones=2):
    '''
    Extract the skin tone (RGB values) from a list of RGB values of the skin.
    
    :param skin_rgb: an array of RGB values of skin
    :type skin_rgb: numpy.ndarray
    
    :param method: the method to extract the skin tones
    :type method: str
    
    :param num_tones: number of top skin tones to extract, only applicable if 
        `method = 'knn'`
    :type num_tones: int
    
    :return skin_tones: the RGB value(s) of the skin tone of the input skin rgb
        dimension `(num_tones, 3)`.
    :rtype skin_tones: numpy.ndarray
    
    Example:
    ```
    >>> skin_rgb = np.array([[148, 128, 121],
                             [144, 124, 117],
                             [142, 121, 116],
                             [106,  75,  73],
                             [106,  75,  73],
                             [111,  81,  83]])
    >>> get_skin_tone(skin_rgb, method='knn', num_tones=2)
    
    [[107,  77,  76],
     [144, 124, 118]]
     
    >>> skin_rgb = np.array([[148, 128, 121],
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
        
def get_skin_tones(image_filepath, method='knn', num_tones=2):
    '''
    Extract the skin tone (RGB values) from an image containing at least one face.
    
    :param image_filepath: filepath to image
    :type image_filepath: str
    
    :param method: the method to extract the skin tones
    :type method: str
    
    :param num_tones: number of top skin tones to extract, only applicable if 
        `method = 'knn'`
    :type num_tones: int
    
    :return skin_tones: the RGB value(s) of the skin tone of the input skin rgb
        dimension `(num_tones, 3)`.
    :rtype skin_tones: numpy.ndarray
    
    :return skin_img: an image showing only the skin, the rest of the non-skin areas of the 
        image are in white.
    :rtype skin_img: PIL.Image.Image
    '''
    skin_colours, skin_img = get_skin_colours(image_filepath)
    
    if skin_colours is None:
        skin_tones = None
    else:
        skin_tones = _get_skin_tones(skin_colours, method=method, num_tones=num_tones)
    
    return skin_tones, skin_img

def distance(embeddings1, embeddings2, distance_metric='cosine'):
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

def _get_face_similarity(embedding_1, embedding_2):
    '''
    Compute a similarity score of 2 face embedding
    1 being exactly the same face, 0 being totally different.

    `None` will be returned if either of the images do not 
    contain at least one face.

    :param embedding_1: embedding vector of face 1
    :type embedding_1: numpy.ndarray

    :param embedding_2: embedding vector of face 1
    :type embedding_2: numpy.ndarray

    :return face_similarity: a similarity score of the faces in the images given.
    :rtype face_similarity: `float` if exactly 1 face is detected in each image, else `None`.
    '''
    
    if embedding_1 is None or embedding_2 is None:
        return None

    face_similarity = 1 - distance(embedding_1, embedding_2, distance_metric='cosine')
    return face_similarity

def get_face_similarity(image_path_1, image_path_2):
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
    
    embedding_1 = get_face_embedding(image_path_1)
    embedding_2 = get_face_embedding(image_path_2)

    face_similarity = _get_face_similarity(embedding_1, embedding_2)
    
    return face_similarity

def _get_skin_tone_similarity(skin_tones_1, skin_tones_2, method="mean"):
    '''
    Obtain the percentage similarity between 2 skin tones.
    A score between 0 to 1 is returned, 0 meaning totally dissimilar,
    1 meaning exactly the same.
    
    A research paper about the best distance metric of two skin rgb values can be found in: 
    http://dx.doi.org/10.5121/csit.2013.3210.
    
    :param skin_tones_1: rgb value(s) of skin tone 1 with dimension `(num_skin_tones, 3)`
    :type skin_tones_1: numpy.ndarray
    
    :param skin_tones_2: rgb value(s) of skin tone 2 with dimension `(num_skin_tones, 3)`
    :type skin_tones_2: numpy.ndarray
    
    :return skin_tone_similarity: percentage similarity between 2 skin tones
    :rtype skin_tone_similarity: float
    '''
    
    assert skin_tones_1.shape == skin_tones_2.shape
    num_skin_tones = len(skin_tones_1)
    
    skin_tone_similarities_sum = 0
    
    for i in range(num_skin_tones):
        abs_diff = np.abs(skin_tones_1[i] - skin_tones_2[i])
        skin_tone_similarities_sum += 1 - np.mean(abs_diff / 255)
    
    skin_tone_similarity = skin_tone_similarities_sum / num_skin_tones
    
    return skin_tone_similarity

def get_skin_tone_similarity(image_filepath_1, image_filepath_2):
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
    
    :return skin_tones_1: rgb values of skin tones of `image_path_1`
    :rtype skin_tones_1: numpy.ndarray
    
    :return skin_tones_2: rgb values of skin tones of `image_path_2`
    :rtype skin_tones_2: numpy.ndarray
    '''
    
    skin_tones_1, skin_1 = get_skin_tones(image_filepath_1, method='knn', num_tones=2)
    skin_tones_2, skin_2 = get_skin_tones(image_filepath_2, method='knn', num_tones=2)
    
    skin_tone_similarity = _get_skin_tone_similarity(skin_tones_1, skin_tones_2)
    
    return skin_tone_similarity, skin_1, skin_2, skin_tones_1, skin_tones_2

def hash_image(image_filepath):
    '''
    Generate a unique identifier of an image.

    :param filepath_to_image: filepath to the image to be hashed
    :type filepath_to_image: str

    :return hash_of_image: a sha256 hash of the image
    :rtype hash_of_image: str
    '''
    
    image = Image.open(image_filepath).convert("RGB")
    image_array = np.asarray(image)
    image_bytes = image_array.data.tobytes()
    
    hash_of_image = hashlib.sha256(image_bytes).hexdigest()
        
    return hash_of_image

def store_face_features_single(pickle_filepath, image_filepath):
    '''
    Store the facial features in the image at the given filepath.
    
    :param pickle_filepath: filepath to the pickle file that stores the facial features
    :type pickle_filepath: str
    
    :param image_filepath: filepath to an image that might contain a face.
    :type image_filepath: str
    
    :return features_dict: a dict that has hash of the image as key and
        facial features of the image as values.
    :rtype features_dict: dict
    '''
    
    # if pickle file not found, create empty dictionary
    if not os.path.exists(pickle_filepath):
        face_features_dict = {}
    else:
        face_features_dict = pickle.load(open(pickle_filepath, "rb"))
    
    # generate unique identifier of the image
    image_hash = hash_image(image_filepath)
    
    # if the image is not already processed
    if image_hash not in face_features_dict:
        embedding = get_face_embedding(image_filepath)
        skin_tones, _ = get_skin_tones(image_filepath)

        face_features_dict[image_hash] = {"embedding": embedding,
                                          "skin_tones": skin_tones,
                                          "image_filepath": os.path.abspath(image_filepath)}
    
    pickle.dump(face_features_dict, open(pickle_filepath, "wb"))
    
    return face_features_dict

def store_face_features_multiple(pickle_filepath, image_directory):
    '''
    Store the facial features for the images in the image directory.
    
    :param pickle_filepath: filepath to the pickle file that stores the facial features
    :type pickle_filepath: str
    
    :param image_directory: filepath to a directory containing images.
        The directory will be recursively walked through.
    :type image_directory: str
    
    :return features_dict: a dict that has hash of the image as key and
        facial features of the image as values.
    :rtype features_dict: dict
    '''
    image_filepaths = []
    
    # recursively walk through the entire image directory
    for root, subdirs, files in os.walk(image_directory):
        for filename in files:
            # if the file is an image, store the face features
            if filename.split('.')[-1] in ['jpg', 'jpeg', 'png']:
                image_filepaths.append(root + '/' + filename)
    
    # for all the images in the image directory, store the face_features
    with tqdm(total=len(image_filepaths)) as progress_bar:
        for image_filepath in image_filepaths:
            progress_bar.set_description('Processing images')
            features_dict = store_face_features_single(pickle_filepath, image_filepath)
            progress_bar.update(1)
            
    
    
    return features_dict

def delete_face_features(pickle_filepath, image_directory):
    '''
    Delete all the face features in the `pickle_filepath` but not in the `image_directory`.
    '''
    
    # if pickle file not found, return None
    if not os.path.exists(pickle_filepath):
        return None
    
    # store image hashes of image in `image_directory`
    image_hashes = []
    
    # recursively walk through the entire image directory
    for root, subdirs, files in os.walk(image_directory):
        for filename in files:
            # if the file is an image, store the image hash
            if filename.split('.')[-1] in ['jpg', 'jpeg', 'png']:
                image_hash = hash_image(root + '/' + filename)
                image_hashes.append(image_hash)
    
    face_features_dict = pickle.load(open(pickle_filepath, "rb"))
    
    for image_hash in face_features_dict.keys():
        if image_hash not in image_hashes:
            face_features_dict.pop(image_hash)
    
    pickle.dump(face_features_dict, open(pickle_filepath, "wb"))
    
    return face_features_dict

def get_top_similar_faces(image_filepath, pickle_filepath, image_directory, top_n=5):
    '''
    Obtain the top few most similar faces
    
    :return most_similar_faces: the filepaths and similarity scores of the most similar faces
    :rtype most_similar_faces: list of dict
    '''
    
    most_similar_faces = []
    
    _ = delete_face_features(pickle_filepath, image_directory)
    face_features_dict = store_face_features_multiple(pickle_filepath, image_directory)
    
    target_image_hash = hash_image(image_filepath)
    target_embedding = face_features_dict[target_image_hash]["embedding"]
    
    if target_embedding is None:
        print(f"No face(s) detected in {image_filepath}")
        return most_similar_faces
    
    for image_hash in face_features_dict:
        if image_hash == target_image_hash:
            continue
            
        embedding = face_features_dict[image_hash]["embedding"]
        
        if embedding is None:
            continue
            
        face_similarity = _get_face_similarity(embedding, target_embedding)
        
        most_similar_faces.append({'face_similarity_score': face_similarity,
                                   'image_filepath': face_features_dict[image_hash]['image_filepath']})
        
        most_similar_faces = sorted(most_similar_faces, 
                                    key= lambda item: item['face_similarity_score'],
                                    reverse=True)
        most_similar_faces = most_similar_faces[:top_n]
    
    return most_similar_faces
    
    
def get_top_similar_skins(image_filepath, pickle_filepath, image_directory, top_n=5):
    '''
    Obtain the top few most similar skins
    
    :return most_similar_faces: the filepaths and similarity scores of the most similar skins
    :rtype most_similar_faces: list of dict
    '''
    
    most_similar_faces = []
    
    _ = delete_face_features(pickle_filepath, image_directory)
    face_features_dict = store_face_features_multiple(pickle_filepath, image_directory)
    
    target_image_hash = hash_image(image_filepath)
    target_skin_tones = face_features_dict[target_image_hash]["skin_tones"]
    
    if target_skin_tones is None:
        print(f"No face(s) detected in {image_filepath}")
        return most_similar_faces
    
    for image_hash in face_features_dict:
        skin_tones = face_features_dict[image_hash]["skin_tones"]
        
        if image_hash == target_image_hash or skin_tones is None:
            continue
            
        skin_similarity = _get_skin_tone_similarity(skin_tones, target_skin_tones)
        
        most_similar_faces.append({'skin_similarity_score': skin_similarity,
                                   'image_filepath': face_features_dict[image_hash]['image_filepath']})
        
        most_similar_faces = sorted(most_similar_faces, 
                                    key= lambda item: item['skin_similarity_score'],
                                    reverse=True)
        most_similar_faces = most_similar_faces[:top_n]
    
    return most_similar_faces

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