from find_boxes import slide_window, SCALE_FACTORS, resize, WINDOW
import cv2
import numpy as np
from moviepy.editor import VideoFileClip


def process(image):
	video_output = 'sliding-window.mp4'
	f = cv2.VideoWriter_fourcc('X','V','I','D')
	video_out = cv2.VideoWriter(video_output, f, 25, (1280,720), True)

	for (scale, y) in SCALE_FACTORS:
		resized_image = resize(image, scale)

		for x in slide_window(resized_image):
			cp = np.copy(resized_image)
			frame = np.zeros_like(image)
			cv2.rectangle(cp, (x, y), (x + WINDOW, y + WINDOW), (0, 255, 0), 2)
			frame[0:cp.shape[0], 0:cp.shape[1], :] = cp
			video_out.write(frame)
			cv2.imshow('f', frame)
			cv2.waitKey(0)

	



if __name__ == '__main__':
	process(cv2.imread('images/test1.jpg'))