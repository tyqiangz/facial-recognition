import argparse

from utils import *

parser = argparse.ArgumentParser(description="""Given an image, return the top few similar faces based on facial similarity and skin tone similarity.

The filenames of the image and the similarity score should be returned.
""",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("IMAGE_PATH_1", help="absolute path to the first image")
parser.add_argument("IMAGE_PATH_2", help="absolute path to the second image")
parser.add_argument("-s", "--show_faces", action="store_true", 
                    help="display the face(s) found in `IMAGE_PATH_1` and `IMAGE_PATH_2`")

args = parser.parse_args()

face_similarity = get_face_similarity(args.IMAGE_PATH_1, args.IMAGE_PATH_2, method="facenet")
skin_tone_similarity, skin_1, skin_2, skin_tone_1, skin_tone_2 = get_skin_tone_similarity_face(args.IMAGE_PATH_1, args.IMAGE_PATH_2)

print("Face Similarity:", face_similarity)
print("Skin Similarity:", skin_tone_similarity)
    
if args.show_faces:
    plot_faces(args.IMAGE_PATH_1)
    plot_faces(args.IMAGE_PATH_2)