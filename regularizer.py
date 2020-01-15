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
	kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(a)
	left_side = 0
	if np.min(a) < 4:
		left_side = -5
	s = linspace(left_side, 1.4 * np.max(a))
	e = kde.score_samples(s.reshape(-1, 1))
	mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
	#print("Minima:", s[mi])
	#print("Maxima:", s[ma])
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
	array_ma = np.rint(array_ma).astype(int)
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


def find_boundaries(array_list, dim_list, bDebug):
	#sort_index = np.argsort(array_list)
	#sorted_array_list = array_list[sort_index]
	#sorted_dim_list = dim_list[sort_index]
	array_list = np.array(array_list)
	dim_list = np.array(dim_list)
	min_val = np.min(array_list)
	max_val = np.max(array_list + dim_list)
	if bDebug:
		print(min_val, max_val)
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
			boundary_list.append(i + min_val)
		if flag_list[i] == 0 and flag_list[i + 1] == 1:
			boundary_list.append(i + 1 + min_val)
	if bDebug:
		print(boundary_list)
	return boundary_list


def spacing_test(left_list, top_list, width_list, height_list, bDebug):
	# simply average the spacing row by row or col by col
	top_align = list(set(top_list))
	left_align = list(set(left_list))
	if bDebug:
		print(top_align)
	new_left_list = []
	new_top_list = []
	new_width_list = []
	new_height_list = []
	# horizontal spacing
	for i in range(len(top_align)):
		wind_index = np.where(top_list == top_align[i])
		left_row = left_list[wind_index]
		width_row = width_list[wind_index]
		sort_index = np.argsort(left_row)
		sorted_left_row = left_row[sort_index]
		sorted_width_row = width_row[sort_index]
		boundaries = find_boundaries(sorted_left_row, sorted_width_row, bDebug)
		boundaries = np.array(boundaries)
		average_spacing = int(np.sum(boundaries[1::2] - boundaries[::2]) / len(boundaries[::2]))
		'''
		if bDebug:
			print(sorted_left_row)
			print(sorted_width_row)
			print(boundaries[1::2], boundaries[::2])
			print(average_spacing)
		'''
		#
		cur_left = np.min(left_row)
		for j in range(len(left_row)):
			new_left_list.append(cur_left)
			new_width_list.append(sorted_width_row[j])
			cur_left = cur_left + sorted_width_row[j] + average_spacing
	# vertical spacing
	for i in range(len(left_align)):
		wind_index = np.where(left_list == left_align[i])
		top_col = top_list[wind_index]
		height_col = height_list[wind_index]
		sort_index = np.argsort(top_col)
		sorted_top_col = top_col[sort_index]
		sorted_height_col = height_col[sort_index]
		boundaries = find_boundaries(sorted_top_col, sorted_height_col, bDebug)
		boundaries = np.array(boundaries)
		average_spacing = int(np.sum(boundaries[1::2] - boundaries[::2]) / len(boundaries[::2]))
		'''
		if bDebug:
			print(sorted_left_row)
			print(sorted_width_row)
			print(boundaries[1::2], boundaries[::2])
			print(average_spacing)
		'''
		#
		cur_top = np.min(top_col)
		for j in range(len(top_col)):
			new_top_list.append(cur_top)
			new_height_list.append(sorted_height_col[j])
			cur_top = cur_top + sorted_height_col[j] + average_spacing
	if bDebug:
		print(new_left_list)
		print(new_width_list)
		print(new_top_list)
		print(new_height_list)
	return new_left_list, new_width_list, new_top_list, new_height_list


def main(input_folder, output_folder):
	input_images = sorted(os.listdir(input_folder))
	for j in range(len(input_images)):
		input_filename = input_folder + '/' + input_images[j]
		output_filename = output_folder + '/' + input_images[j]
		print(input_filename)
		width, height, left_list, top_list, width_list, height_list = contours_test(input_filename)
		debug = True
		left_out = align_test(left_list, debug)
		top_out = align_test(top_list, debug)
		width_out = align_test(width_list, debug)
		height_out = align_test(height_list, debug)

		new_left_out, new_width_out, new_top_out, new_height_out = spacing_test(left_out, top_out, width_out, height_out, debug)
		drawing = np.zeros((width, height, 3), dtype=np.uint8)
		drawing = drawing + 255
		window_color = (0, 0, 0)
		for i in range(len(left_out)):
			#cv.rectangle(drawing, (left_out[i], top_out[i]), (left_out[i] + width_out[i], top_out[i] + height_out[i]), window_color, -1)
			cv.rectangle(drawing, (new_left_out[i], new_top_out[i]), (new_left_out[i] + new_width_out[i] - 1, new_top_out[i] + new_height_out[i] - 1),
						 window_color, -1)
		# Show in a window
		cv.imwrite(output_filename, drawing)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("input_folder", help="path to input images folder (e.g., input_data)")
	parser.add_argument("output_folder", help="path to output images folder (e.g., input_data)")
	args = parser.parse_args()
	main(args.input_folder, args.output_folder)
	#contours_test("facade_00202.png")
	#kde_test([20, 26, 20, 20, 26, 21, 20, 26, 21, 20, 27, 21])
