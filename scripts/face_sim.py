import argparse

import face_recognition
from PIL import Image, ImageDraw
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument("IMAGE_PATH_1", help="absolute path to the first image")
parser.add_argument("IMAGE_PATH_2", help="absolute path to the second image")
parser.add_argument("-s", "--show_faces", action="store_true", 
                    help="display the face(s) found in `IMAGE_PATH_1` and `IMAGE_PATH_2`")

args = parser.parse_args()

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


face_similarity = calc_face_distance(args.IMAGE_PATH_1, args.IMAGE_PATH_2)

if face_similarity is not None:
    print(face_similarity[0])
else:    
    print(face_similarity)
    
if args.show_faces:
    plot_faces(args.IMAGE_PATH_1)
    plot_faces(args.IMAGE_PATH_2)