#######################################################################
# Generate HTML file that shows the input and output images to compare.

from os.path import isfile, join
from PIL import Image
import os
import argparse
import numpy as np
import json
import subprocess
import sys
import shutil
import glob
import pandas as pd
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from skimage import io
import cv2 as cv
import random as rng
from numpy import array, linspace
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def regularize_test():
	input = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1, 0],[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1, 0]])
	print(input)
	print(input.shape)
	c = input.mean(0).reshape((-1, input.shape[1]))
	r = input.mean(1).reshape((input.shape[0], -1))
	r_2 = np.array(r > 0.33).astype(int)
	c_2 = np.array(c > 0.33).astype(int)
	print(r_2)
	print(c_2)
	print(r_2 * c_2)


def kde_test(array_list):
	a = array(array_list).reshape(-1, 1)
	kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
	left_side = 0
	if np.min(a) < 4:
		left_side = -5
	s = linspace(left_side, 1.4 * np.max(a))
	e = kde.score_samples(s.reshape(-1, 1))
	mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
	#print("Minima:", s[mi])
	print("Maxima:", s[ma])
	#plt.plot(s, e)
	#plt.show()
	return s[mi], s[ma]


def contours_test(img_filename):
	src_gray = cv.imread(img_filename, cv.IMREAD_UNCHANGED)
	rng.seed(12345)
	# Find contours
	_, contours, hierarchy = cv.findContours(src_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	# Draw contours
	drawing = np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)
	number_windows = 0
	boundRect = [None] * len(contours)
	top_list = []
	left_list = []
	height_list = []
	width_list = []
	for i in range(len(contours)):
		boundRect[i] = cv.boundingRect(contours[i])
		if hierarchy[0][i][3] != 0:
			continue
		left_list.append(boundRect[i][0])
		top_list.append(boundRect[i][1])
		width_list.append(boundRect[i][2])
		height_list.append(boundRect[i][3])
		color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
		#cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
		#cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
		number_windows = number_windows + 1
	#print(number_windows)
	# Show in a window
	#cv.imshow('Contours', drawing)
	#cv.waitKey()
	return src_gray.shape[0], src_gray.shape[1], left_list, top_list, width_list, height_list


def align_test(array_list, bDebug):
	_, array_ma = kde_test(array_list)
	array_ma = array_ma.astype(int)
	if bDebug:
		print("Array List:", array_list)
		print("Array Maxima:", array_ma)
	dis_array1 = np.tile(array_ma, (len(array_list), 1))
	dis_array2 = np.transpose(np.tile(array_list, (len(array_ma), 1)))
	#print(dis_array1)
	#print(dis_array2)
	pos = np.argmin(np.abs(dis_array1 - dis_array2), axis=1).reshape(-1, 1)
	array_out = array_ma[pos].reshape(1, -1).flatten()
	if bDebug:
		print("Array out:", array_out)
	return array_out


def spacing(array_list, dim_list):
	#sort_index = np.argsort(array_list)
	#sorted_array_list = array_list[sort_index]
	#sorted_dim_list = dim_list[sort_index]
	min_val = np.min(array_list)
	max_val = np.max(array_list + dim_list)
	# compute flag_list
	flag_list = []
	for i in range(min_val, max_val + 1):
		flag = 0
		for j in range(len(array_list)):
			if array_list[j] <= i <= array_list[j] + dim_list[j]:
				flag = 1
				break
		flag_list.append(flag)
	# find boundaries
	boundary_list = []
	for i in range(len(flag_list) - 1):
		if flag_list[i] == 1 and flag_list[i + 1] == 0:
			boundary_list.append(i)
		if flag_list[i] == 0 and flag_list[i + 1] == 1:
			boundary_list.append(i + 1)
	return boundary_list


def main(input_folder, output_folder):
	input_images = sorted(os.listdir(input_folder))
	for j in range(len(input_images)):
		input_filename = input_folder + '/' + input_images[j]
		output_filename = output_folder + '/' + input_images[j]
		width, height, left_list, top_list, width_list, height_list = contours_test(input_filename)
		debug = True
		left_out = align_test(left_list, debug)
		top_out = align_test(top_list, debug)
		width_out = align_test(width_list, debug)
		height_out = align_test(height_list, debug)
		drawing = np.zeros((width, height, 3), dtype=np.uint8)
		drawing = drawing + 255
		window_color = (0, 0, 0)
		for i in range(len(left_out)):
			cv.rectangle(drawing, (left_out[i], top_out[i]), (left_out[i] + width_out[i] - 1, top_out[i] + height_out[i] - 1), window_color, -1)
		# Show in a window
		cv.imwrite(output_filename, drawing)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("input_folder", help="path to input images folder (e.g., input_data)")
	parser.add_argument("output_folder", help="path to output images folder (e.g., input_data)")
	args = parser.parse_args()
	main(args.input_folder, args.output_folder)
	#contours_test("facade_00202.png")
