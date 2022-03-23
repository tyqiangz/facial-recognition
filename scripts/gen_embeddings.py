import os
from PIL import Image
import hashlib



def hash_image(filepath_to_image):
    '''
    Generate a unique identifier of an image.

    :param filepath_to_image: filepath to the image to be hashed
    :type filepath_to_image: str

    :return hash_of_image: a sha256 hash of the image
    :rtype hash_of_image: str
    '''
    with open(filepath_to_image, "rb") as f:
        hash_of_image = hashlib.sha256(f.read()).hexdigest()
        
        return hash_of_image

