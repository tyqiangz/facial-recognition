import argparse
from utils import *
from datetime import datetime
import pprint
import os

parser = argparse.ArgumentParser(description="""Given an image, return the top few similar faces based on facial similarity and skin tone similarity.

The filenames of the image and the similarity score should be returned.
""",
                                 formatter_class=argparse.RawTextHelpFormatter)

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive integer value" % value)
    return ivalue

parser.add_argument("IMAGE_PATH", help="absolute path to an image")
parser.add_argument("NUM_FACES_TO_RETURN", 
                    help="number of most similar faces to return, must be a positive integer value",
                    type=check_positive)

args = parser.parse_args()

top_similar_faces = get_top_similar_faces(image_filepath=args.IMAGE_PATH, 
                                          image_directory='../images',
                                          pickle_filepath='../backend/facial_features.pickle',
                                          top_n=int(args.NUM_FACES_TO_RETURN))
top_similar_skins = get_top_similar_skins(image_filepath=args.IMAGE_PATH, 
                                          image_directory='../images',
                                          pickle_filepath='../backend/facial_features.pickle',
                                          top_n=int(args.NUM_FACES_TO_RETURN))

results_filename = 'results_' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '.txt' 
results_filepath = '../results/' + results_filename

with open(results_filepath, 'w') as f:
    f.write(f'Most similar faces for {args.IMAGE_PATH}\n\n')
    f.writelines(pprint.pformat(top_similar_faces) + '\n')
    f.write('-'*100 + '\n')
    f.write(f'Most similar skin tone for {args.IMAGE_PATH}\n\n')
    f.writelines(pprint.pformat(top_similar_skins))

print(f'Results are stored in {os.path.abspath(results_filepath)}')