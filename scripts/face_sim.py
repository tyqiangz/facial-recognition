import argparse

from utils import *

parser = argparse.ArgumentParser(description="""Given two images, detect the face(s) and compute a similarity score for the faces.

If each image has 1 and only 1 face, then a score ranging from 0 to 1 is computed.
0 being the faces are extremely different, 1 being the faces are exactly the same.

If any of the two images has no faces detected or more than 1 faces detected, a value None is returned.
""",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("IMAGE_PATH_1", help="absolute path to the first image")
parser.add_argument("IMAGE_PATH_2", help="absolute path to the second image")
parser.add_argument("-s", "--show_faces", action="store_true", 
                    help="display the face(s) found in `IMAGE_PATH_1` and `IMAGE_PATH_2`")

args = parser.parse_args()

face_similarity = calc_face_distance(args.IMAGE_PATH_1, args.IMAGE_PATH_2)

if face_similarity is not None:
    print(face_similarity[0])
else:    
    print(face_similarity)
    
if args.show_faces:
    plot_faces(args.IMAGE_PATH_1)
    plot_faces(args.IMAGE_PATH_2)