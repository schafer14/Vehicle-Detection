import pickle
import cv2
import numpy as np
import time
from skimage.feature import hog
import math
from moviepy.editor import VideoFileClip

MIN_PROB = .95
SCALE_FACTORS = [(0.7, 280), (0.6, 240), (0.5, 196), (0.5, 210), (0.4, 170), (0.4, 145), (0.3, 130)]
REPEAT_BONUS = .9
DIST_FOR_BONUS = 50
orient = 9
pixels_per_cell = (8, 8)
cell_per_block = (2, 2)
WINDOW = 64
nblocks_per_window = (WINDOW // pixels_per_cell[0]) - cell_per_block[0] + 1
default_x = 1280

def extract_features(img):
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h1 = hog(HSV[:,:,0], orientations=orient, pixels_per_cell=pixels_per_cell, transform_sqrt=True, cells_per_block=cell_per_block)
    h2 = hog(HSV[:,:,1], orientations=orient, pixels_per_cell=pixels_per_cell, transform_sqrt=True, cells_per_block=cell_per_block)
    h3 = hog(HSV[:,:,2], orientations=orient, pixels_per_cell=pixels_per_cell, transform_sqrt=True, cells_per_block=cell_per_block)
#     channel1_hist, _ = np.histogram(img[:,:,0], bins=32)
#     channel2_hist, _ = np.histogram(img[:,:,1], bins=32)
#     channel3_hist, _ = np.histogram(img[:,:,2], bins=32)
    
    return np.concatenate((h1, h2, h3, HSV[:, :, 1].ravel()))

# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, pix=20):
	x = 0
	coord = []

	while x + WINDOW < img.shape[1]:
		coord.append(x)
		x += pix

	return coord

# remove overlapping windows from image
def non_max_suppression(boxes, probs):
	overlapThresh = 0.1
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this is important since
	# we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and sort the bounding boxes by their associated
	# probabilities
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value to the list of
		# picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding box and the
		# smallest (x, y) coordinates for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater than the
		# provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")

# Calculate the distance between to points
def dist(p1, p2):
	dx = p1[0] - p2[0]
	dy = p1[1] - p2[1]

	return dx * dx + dy * dy

#  Resizes the image
def resize(img, size):
	return cv2.resize(img, (int(img.shape[1] * size), int(img.shape[0] * size)))

# Loading the saved model pickle
f = open('classifier.pkl', 'rb')
model = pickle.load(f)
f = open('normalizer.pkl', 'rb')
X_scaler = pickle.load(f)
centroids = []

def process(image):
	# The centroids need to be stored from one pass to the next. So externally to this function.
	# A closure might have been a better data structure then defining a gloabal variable.
	global centroids
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	probs = []
	boxes = []
	for (scale, y) in SCALE_FACTORS:
		resized_image = resize(image, scale)

		for x in slide_window(resized_image):
			cp = np.copy(resized_image)
			
			roi = cp[y:y + WINDOW, x:x + WINDOW]

			# cv2.rectangle(cp, (x, y), (x + WINDOW, y + WINDOW), (0, 255, 0), 2)
			# cv2.imshow('rect', cp)
			# cv2.waitKey(25)

			features = extract_features(roi)			

			normal = X_scaler.transform([features])
			prob = model.predict_proba(normal)[0][1]
			centroid = (x, y)
			need_prob = MIN_PROB

			if any([centroid for centroid in centroids if dist((x, y), centroid) < DIST_FOR_BONUS]):
				need_prob -= REPEAT_BONUS

			centroids = []
			if prob > need_prob:
				centroids.append(centroid)
				
				box = []
				box.append(math.floor(x / scale))
				box.append(math.floor(y / scale))
				box.append(math.floor((x + WINDOW) / scale))
				box.append(math.floor((y + WINDOW) / scale))

				boxes.append(box)
				probs.append(prob)


	singles = non_max_suppression(np.array(boxes), probs)

	for box in singles:
		cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

	return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


#  # Video
if __name__ == '__main__':
	video_output = './videos/result.mp4'
	clip1 = VideoFileClip("./videos/project_video.mp4")
	white_clip = clip1.fl_image(process)
	white_clip.write_videofile(video_output, audio=False)
