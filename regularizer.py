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
from itertools import groupby
import statistics
from sklearn.cluster import MeanShift

w_a = 0.01
w_ss = 0.03
w_sp = 0.01

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


def kde_1d(array_list):
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


def mean_shift(array_list_w, array_list_h, bDebug):
	x = array(array_list_w)
	y = array(array_list_h)
	data = np.vstack([x, y]).T
	# define the model
	model = MeanShift(bandwidth=5)
	# fit model and predict clusters
	yhat = model.fit_predict(data)
	# retrieve unique clusters
	clusters = np.unique(yhat)
	# create scatter plot for samples from each cluster
	groups = []
	for cluster in clusters:
		# get row indexes for samples with this cluster
		row_ix = np.where(yhat == cluster)
		# create scatter of these samples
		groups.append(data[row_ix])
	if bDebug:
		print("Groups:", groups)
	return groups


def contours_test(img_filename):
	src_gray = cv.imread(img_filename, cv.IMREAD_UNCHANGED)
	#print(src_gray.shape)
	#if src_gray.shape[2] == 4:
		#src_gray = cv.cvtColor(src_gray, cv.COLOR_BGRA2GRAY)
	rng.seed(12345)
	# Find contours
	_, thresh = cv.threshold(src_gray, 127, 255, 0)
	_, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	#_, contours, hierarchy = cv.findContours(src_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	#print(len(contours))
	# Draw contours
	#drawing = np.zeros((src_gray.shape[0], src_gray.shape[1], 3), dtype=np.uint8)
	number_windows = 0
	boundRect = [None] * len(contours)
	top_list = []
	bot_list = []
	left_list = []
	right_list = []
	height_list = []
	width_list = []
	for i in range(len(contours)):
		boundRect[i] = cv.boundingRect(contours[i])
		left_list.append(boundRect[i][0])
		top_list.append(boundRect[i][1])
		right_list.append(boundRect[i][0] + boundRect[i][2] - 1)
		bot_list.append(boundRect[i][1] + boundRect[i][3] - 1)
		width_list.append(boundRect[i][2])
		height_list.append(boundRect[i][3])
		#color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
		#cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
		#cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 1)
		number_windows = number_windows + 1
	#print(number_windows)
	# Show in a window
	#cv.imshow('Contours', drawing)
	#cv.waitKey()
	return src_gray.shape[0], src_gray.shape[1], left_list, right_list, top_list, bot_list, width_list, height_list


def split2groups(array_list, bDebug):
	_, array_ma = kde_1d(array_list)
	array_ma = np.rint(array_ma).astype(int)
	if bDebug:
		print("Array List:", array_list)
		print("Array Maxima:", array_ma)
	dis_array1 = np.tile(array_ma, (len(array_list), 1))
	dis_array2 = np.transpose(np.tile(array_list, (len(array_ma), 1)))
	# print(dis_array1)
	# print(dis_array2)
	pos = np.argmin(np.abs(dis_array1 - dis_array2), axis=1).reshape(-1, 1)
	# array_out = array_ma[pos].reshape(1, -1).flatten()
	list_one = pos.reshape(1, -1).flatten()
	list_two = array_list
	# print('list_one is ', list_one)
	# print('list_two is ', list_two)
	data = zip(list_one, list_two)
	groups = []
	for k, g in groupby(data, lambda x: x[0]):
		values = [n for m, n in list(g)]
		groups.append(values)  # Store group iterator as a list
	if bDebug:
		print("Groups:", groups)
	return groups


def alignment_error(array_list, width_list_sorted, bDebug):
	groups = split2groups(array_list, bDebug)
	if bDebug:
		print("width list is ", width_list_sorted)
	# compute score
	std_groups = 0
	current_index = 0
	for i in range(len(groups)):
		min_width = np.min(width_list_sorted[current_index: current_index + len(groups[i])])
		std_groups = std_groups + np.std(groups[i]) / min_width
		if bDebug:
			print(width_list_sorted[current_index: current_index + len(groups[i])])
			print(np.std(groups[i]))
		current_index = current_index + len(groups[i])
	# encourage fewer and larger groups
	error =  std_groups + w_a * len(groups)
	if bDebug:
		print("Std_groups:", std_groups)
		print("Align error:", error)
	return error


def same_size_error(array_list_w, array_list_h, bDebug):
	groups = mean_shift(array_list_w, array_list_h, bDebug)
	# compute error
	std_groups = 0
	for i in range(len(groups)):
		mean_p = np.mean(groups[i], axis=0)
		distance = 0
		# min height or width
		min_v = 1000
		for j in range(len(groups[i])):
			distance = distance + np.square(np.linalg.norm(groups[i][j]-mean_p))
			min_v = min([groups[i][j][0], groups[i][j][1], min_v])
		distance = np.sqrt(distance) / (len(groups[i]))
		std_groups = std_groups + distance / min_v
		if bDebug:
			print(distance)
			print(min_v)
	# encourage fewer and larger groups
	error = std_groups + w_ss * len(groups)
	if bDebug:
		print("Std_groups:", std_groups)
		print("Same Size error:", error)
	return error


def split(array_list, bDebug):
	_, array_ma = kde_1d(array_list)
	array_ma = np.rint(array_ma).astype(int)
	if bDebug:
		print("Array List:", array_list)
		print("Array Maxima:", array_ma)
	dis_array1 = np.tile(array_ma, (len(array_list), 1))
	dis_array2 = np.transpose(np.tile(array_list, (len(array_ma), 1)))
	# print(dis_array1)
	# print(dis_array2)
	pos = np.argmin(np.abs(dis_array1 - dis_array2), axis=1).reshape(-1, 1)
	array_out = array_ma[pos].reshape(1, -1).flatten()
	return array_out


def hor_spacing(left_list, right_list, top_list, bot_list, debug):
	top_list = split(top_list, debug)
	top_align = np.unique(top_list)
	#print(top_align)
	#print('left_list ', left_list)
	#print('right_list ', right_list)
	# horizontal spacing
	left_array = np.array(left_list)
	right_array = np.array(right_list)
	spacings = []
	for i in range(len(top_align)):
		wind_index = np.where(top_list == top_align[i])
		#print(wind_index)
		left_row = left_array[wind_index]
		right_row = right_array[wind_index]
		sort_index = np.argsort(left_row)
		sorted_left_row = left_row[sort_index]
		sorted_right_row = right_row[sort_index]
		#print(sorted_left_row)
		#print(sorted_right_row)
		#print(sorted_left_row[1::1])
		#print(sorted_right_row[:len(sorted_right_row) - 1])
		spacings.extend(sorted_left_row[1::1] - sorted_right_row[:len(sorted_right_row) - 1])
	if debug:
		print(spacings)
	return spacings


def ver_spacing(left_list, right_list, top_list, bot_list, debug):
	left_list = split(left_list, debug)
	left_align = np.unique(left_list)
	# print(left_align)
	# print('top_list ', top_list)
	# print('bot_list ', bot_list)
	# vertical spacing
	top_array = np.array(top_list)
	bot_array = np.array(bot_list)
	spacings = []
	for i in range(len(left_align)):
		wind_index = np.where(left_list == left_align[i])
		# print(wind_index)
		top_row = top_array[wind_index]
		bot_row = bot_array[wind_index]
		sort_index = np.argsort(top_row)
		sorted_top_row = top_row[sort_index]
		sorted_bot_row = bot_row[sort_index]
		# print(sorted_top_row)
		# print(sorted_bot_row)
		# print(sorted_top_row[1::1])
		# print(sorted_bot_row[:len(sorted_bot_row) - 1])
		spacings.extend(sorted_top_row[1::1] - sorted_bot_row[:len(sorted_bot_row) - 1])
	if debug:
		print(spacings)
	return spacings


def same_spacing_error(left_list, right_list, top_list, bot_list, debug):
	ver_error = 0
	hor_error = 0

	hor_spacings = hor_spacing(left_list, right_list, top_list, bot_list, debug)
	sorted_hor_spacings = sorted(hor_spacings)
	hor_groups = split2groups(sorted_hor_spacings, debug)

	# compute score
	std_groups = 0.0
	for i in range(len(hor_groups)):
		min_v = np.min(hor_groups[i])
		std_groups = std_groups + np.std(hor_groups[i]) / min_v
		print(hor_groups[i])
		if debug:
			print(np.std(np.array(hor_groups[i])))
	# encourage fewer and larger groups
	hor_error = std_groups + w_sp * len(hor_groups)
	if debug:
		print("Hor Std_groups:", std_groups)
		print("Hor Align error:", hor_error)
	

	ver_spacings = ver_spacing(left_list, right_list, top_list, bot_list, debug)
	sorted_ver_spacings = sorted(ver_spacings)
	ver_groups = split2groups(sorted_ver_spacings, debug)

	# compute score
	std_groups = 0.0
	for i in range(len(ver_groups)):
		min_v = np.min(ver_groups[i])
		std_groups = std_groups + np.std(ver_groups[i]) / min_v
		print(ver_groups[i])
		if debug:
			print(np.std(np.array(ver_groups[i])))
	# encourage fewer and larger groups
	ver_error = std_groups + w_sp * len(ver_groups)
	if debug:
		print("Ver Std_groups:", std_groups)
		print("Ver Align error:", ver_error)
	return hor_error, ver_error


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
		print(flag_list)
		print(boundary_list)
	return boundary_list


def horz_spacing_test(left_list, top_list, width_list, height_list, bDebug):
	# simply average the spacing row by row or col by col
	top_align = list(set(top_list))
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
		top_col = top_list[wind_index]
		height_col = height_list[wind_index]
		sort_index = np.argsort(left_row)
		sorted_left_row = left_row[sort_index]
		sorted_width_row = width_row[sort_index]
		sorted_top_col = top_col[sort_index]
		sorted_height_col = height_col[sort_index]
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
			new_top_list.append(sorted_top_col[j])
			new_height_list.append(sorted_height_col[j])
			cur_left = cur_left + sorted_width_row[j] + average_spacing
	return new_left_list, new_width_list, new_top_list, new_height_list


def vert_spacing_test(left_list, top_list, width_list, height_list, bDebug):
	# simply average the spacing row by row or col by col
	left_align = list(set(left_list))
	if bDebug:
		print(left_list)
	new_left_list = []
	new_top_list = []
	new_width_list = []
	new_height_list = []
	# vertical spacing
	for i in range(len(left_align)):
		wind_index = np.where(left_list == left_align[i])
		top_col = top_list[wind_index]
		height_col = height_list[wind_index]
		left_row = left_list[wind_index]
		width_row = width_list[wind_index]
		sort_index = np.argsort(top_col)
		sorted_top_col = top_col[sort_index]
		sorted_height_col = height_col[sort_index]
		sorted_left_row = left_row[sort_index]
		sorted_width_row = width_row[sort_index]
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
			new_left_list.append(sorted_left_row[j])
			new_width_list.append(sorted_width_row[j])
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
		width, height, left_list, right_list, top_list, bot_list, width_list, height_list = contours_test(input_filename)

		# sort list
		# Alignment Regularization

		'''
		debug = False
		sort_index_left = np.argsort(left_list)
		left_list_sorted = np.array(left_list)[sort_index_left]
		width_list_sorted = np.array(width_list)[sort_index_left]
		left_align_error = alignment_error(left_list_sorted, width_list_sorted, debug)

		sort_index_right = np.argsort(right_list)
		right_list_sorted = np.array(right_list)[sort_index_right]
		width_list_sorted = np.array(width_list)[sort_index_right]
		right_align_error = alignment_error(right_list_sorted, width_list_sorted, debug)
		
		sort_index_top = np.argsort(top_list)
		top_list_sorted = np.array(top_list)[sort_index_top]
		height_list_sorted = np.array(height_list)[sort_index_top]
		top_align_error = alignment_error(top_list_sorted, height_list_sorted, debug)

		sort_index_bot = np.argsort(bot_list)
		bot_list_sorted = np.array(bot_list)[sort_index_bot]
		height_list_sorted = np.array(height_list)[sort_index_bot]
		bot_align_error = alignment_error(bot_list_sorted, height_list_sorted, debug)

		print('total align error is ', left_align_error + right_align_error + top_align_error + bot_align_error)
		'''

		# Same Size Regularization
		#debug = True
		#size_error = same_size_error(width_list, height_list, debug)
		#print('total size error is ',size_error)


		# Same Spacing Regularization
		debug = False
		hor_error, ver_error = same_spacing_error(left_list, right_list, top_list, bot_list, debug)
		print('total spacing error is ', hor_error +  ver_error)


def add_border(input_folder, output_folder):
	input_images = sorted(os.listdir(input_folder))
	for j in range(len(input_images)):
		input_filename = input_folder + '/' + input_images[j]
		output_filename = output_folder + '/' + input_images[j]
		print(input_filename)
		src_gray = cv.imread(input_filename, cv.IMREAD_UNCHANGED)
		border = 3
		value = (255, 255, 255)
		image = cv.copyMakeBorder(src_gray, border, border, border, border, cv.BORDER_CONSTANT, None, value)
		cv.imwrite(output_filename, image)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("input_folder", help="path to input images folder (e.g., input_data)")
	parser.add_argument("output_folder", help="path to output images folder (e.g., input_data)")
	args = parser.parse_args()
	main(args.input_folder, args.output_folder)
	#add_border(args.input_folder, args.output_folder)
	#contours_test("facade_00202.png")
	#kde_test([20, 26, 20, 20, 26, 21, 20, 26, 21, 20, 27, 21])
