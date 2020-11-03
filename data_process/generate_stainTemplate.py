import argparse
import sys
import os
sys.path.append("img2vec/img2vec_pytorch")
from img_to_vec import Img2Vec
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from shutil import copyfile

'''
Running:
python generate_stainTemplate.py --input_dir ../datasets/Train/Images/ --output_dir ../datasets/Train/Stain_template/
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Stain Normalization')
    parser.add_argument(
        '--input_dir',
        dest='input_dir',
        help='Input Directory for training images',
        default='../datasets/Train/Images/',
        type=str
    )
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        help='Output Directory for Normalized Images',
        default='../datasets/Train/Stain_template/',
        type=str
    )
    parser.add_argument(
        '--ext',
        dest='ext',
        help='Image Extension Type',
        default='png',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

# Create directory is not exists
def mkdir_if_nonexist(path):
	if not os.path.exists(path):
		os.makedirs(path)
def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))

def main(args):
	
	img2vec = Img2Vec()
	file_list = list_files(args.input_dir, args.ext)
	# For each test image, we store the filename and vector as key, value in a dictionary
	pics_vec_list = []
	pics_name = []
	for i, im_file in enumerate(file_list):
		img = Image.open(os.path.join(args.input_dir, im_file))
		vec = img2vec.get_vec(img)
		pics_name.append(im_file)
		pics_vec_list.append(vec)

	print("Do KMeans")
	X = np.array(pics_vec_list)
	kmeans = KMeans(n_clusters=12, random_state=0).fit(X)
	labels = kmeans.labels_
	print(labels)
	cluster_centers = kmeans.cluster_centers_


	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)

	closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
	print(closest)

	for c in range(len(labels_unique)):
		mkdir_if_nonexist(os.path.join(args.output_dir, 'class_' + str(c)))
	
	select_path = os.path.join(args.output_dir, 'class_select')
	mkdir_if_nonexist(select_path)

	print("number of estimated clusters : %d" % n_clusters_)

	for i in range(len(pics_name)):
		cluster_id = labels[i]
		output_fpath = os.path.join(os.path.join(args.output_dir, 'class_' + str(cluster_id)), pics_name[i])
		input_fpath = os.path.join(args.input_dir, pics_name[i])
		copyfile(input_fpath, output_fpath)
	
	for i in range(len(closest)):
		select_pic = pics_name[closest[i]]
		output_fpath = os.path.join(select_path, select_pic)
		input_fpath = os.path.join(args.input_dir, select_pic)
		copyfile(input_fpath, output_fpath)
		
	return 0
		
if __name__ == '__main__':
    args = parse_args()
    main(args)
