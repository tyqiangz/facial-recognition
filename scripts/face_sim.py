import argparse
parser = argparse.ArgumentParser()

parser.add_argument("IMAGE_PATH_1", help="path to the first image")
parser.add_argument("IMAGE_PATH_2", help="path to the second image")

args = parser.parse_args()

print(args.IMAGE_PATH_1)
print(args.IMAGE_PATH_2)