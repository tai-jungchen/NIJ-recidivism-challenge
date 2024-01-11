"""
i-vt
"""
import tobii_research as tr
import time
import numpy as np

last_left = 0, 0, 0
last_right = 0, 0, 0
FREQ = 120


def main():
	found_eyetrackers = tr.find_all_eyetrackers()
	my_eyetracker = found_eyetrackers[0]
	print("Address: " + my_eyetracker.address)
	print("Model: " + my_eyetracker.model)
	print("Name (It's OK if this is empty): " + my_eyetracker.device_name)
	print("Serial number: " + my_eyetracker.serial_number)
	my_eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
	# stop
	time.sleep(3)
	my_eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)


def gaze_data_callback(gaze_data, eye_angle_change_callback):
	# print("Left eye: ({gaze_left_eye}) \t Right eye: ({gaze_right_eye})".format(
	# 	gaze_left_eye=gaze_data['left_gaze_point_on_display_area'],
	# 	gaze_right_eye=gaze_data['right_gaze_point_on_display_area']))

	global last_left, last_right
	# Print gaze points of left and right eye #
	print("Left eye: ({gaze_left_eye}) \t Right eye: ({gaze_right_eye})".format(
		gaze_left_eye=gaze_data['left_gaze_point_in_user_coordinate_system'],
		gaze_right_eye=gaze_data['right_gaze_point_in_user_coordinate_system']))

	# parameters for angular speed #
	left_gaze_origin_2 = gaze_data['left_gaze_origin_in_user_coordinate_system']
	left_gaze_point_1 = last_left
	left_gaze_point_2 = gaze_data['left_gaze_point_in_user_coordinate_system']
	right_gaze_origin_2 = gaze_data['right_gaze_origin_in_user_coordinate_system']
	right_gaze_point_1 = last_right
	right_gaze_point_2 = gaze_data['right_gaze_point_in_user_coordinate_system']

	if last_left == (0, 0, 0):		# first run
		last_left = gaze_data['left_gaze_point_in_user_coordinate_system']
		last_right = gaze_data['right_gaze_point_in_user_coordinate_system']
		print(f'last_left: {last_left}')
	else:
		# print(f"in else loop")
		left_angular_speed = eye_angle_change_callback(left_gaze_origin_2, left_gaze_point_1, left_gaze_point_2)
		right_angular_speed = eye_angle_change_callback(right_gaze_origin_2, right_gaze_point_1, right_gaze_point_2)
		print(f'Left angular speed: {left_angular_speed}\t Right angular speed: {right_angular_speed}')

	# print(f'gaze_data: {gaze_data}')
	# print(gaze_data)


def eye_angle_change(gaze_origin_2, gaze_point_1, gaze_point_2):
	"""
	This function computes the change of eye angle (2 * alpha * f)
	:param gaze_origin_1: gaze origin at time 1
	:param gaze_point_1: gaze point at time 1
	:param gaze_point_2: gaze point at time 2
	:return: 2 * alpha * f
	"""
	gp_dist = np.linalg.norm(gaze_point_1 - gaze_point_2)
	mid_point = (gaze_point_1 + gaze_point_2) / 2
	om_dist = np.linalg.norm(mid_point - gaze_origin_2)
	alpha = np.arctan((gp_dist/2) / om_dist)
	angular_speed = 2 * alpha * FREQ

	print(f'angular_speed: {angular_speed}')

	return angular_speed


if __name__=='__main__':
	main()