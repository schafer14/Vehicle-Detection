# Vehicle Detection

## Result

[Resulting Video](https://youtu.be/K4yqHh9kSn8)

## Goals

To find and track vehicles in driving video

## Features

- Training
	- HOG feature extraction on each HSV color channel with saturation channel raw pixel insities.
	- Hard nagative mining. Increase test data set to 98.5% from validation accuracy of 98.0%
	- Linear SVM model
	- 98.5% accuracy on test data set
	- Training, validation, and test data sets
- Testing
	- Non maximal surpression
	- Vehicle location prediction
	- Updated vehicle detection probability based on previous frames
	- Hand selected sliding window coordinates

 
## Imporvemets

- Use HOG feature extractor once per frame
	- This will increase speed due to the fact that the HOG feature extractor is one of the most expensive computations and is run multiple times per frame unnecessarily
	- The current algorithm uses each HSV color channel for three different HOG extractors. The H and V channel contribute little and could be removed for efficency.
	- The ordered raw pixel intensities of the saturation channel are used as features. Considering these are ordered it is highly unlikely these could help train the model, but in practice seem to be actually helping. These might be able to be removed for efficency. 
	- Color histogram of the saturation channel intuitively would help more then the raw pixel intensities, but in practice this doesn't seem to be the case. 
- Better tracking algorithms
	- Currently tracking is just being used to predict where vehicles are more likely to be given their position in old frames. A more effective algorithm might predict future locations as well as record more then one previous frame and more data then just the location.
	- Tracking class
		- The tracking algorithm is implemented with global variables and should be abstracted to it's own class or closure
- Sliding window
	- To increase effiecency I have hand selected scales and heights of sliding windows. While the selections work pretty well they also introduce "blid spots" when the car is closer to the horizon and two lanes to the left. 
	- Non maximal surpression algorithm could be extended to remove boxes that not only over lapped with the maximal box but were also close by. 

## Procedure

### Training

To train the model I found that the slightly unintuitive combination of the HOG features from each channel of the HSV color image along with the raw pixel intensities of the saturation channel had the best results. One huge problem I faced was the training data set of images was not representative of the video I recieved and this caused for huge descrpencies between the accurcay on the test dataset and the accuracy of the video. 

For training the model I split the training dataset into three subsets (training, validation, and test). I initially trained the model on the training set and tested on the validation set. I then extracted a feature vector from each image

#### The feature vector

I chose HOG features as my main features augemented with the saturation pixel intensities. I used 9 orientation bins for my HOG algorithm as well as 8x8 pixels per cell as these seemed like good default values. I did experiment with using either 1x1 or 2x2 cells per box and found that the 2x2 option was better. 

I spent some time tweaking the augmenting feature. I finally settled on the aforementioned unintitive saturation pixel intensities. Other features that did not work so well was the color histograms from the HSV and grayscale color spaces. 

I also looked at doing both the HOG descriptor and the histogram/intensities using grayscale instead of the HSV color space and found similar results. The grayscale space is probably more efficent because only one HOG algorithm is run, but I arbirarily chose to use the HSV system.

#### Normalization

After the feature vectors were extracted I normalized values to make sure one particularl feature would not dominate the rest. 

#### The Model

I used a linear SVM as my model. I found similar preformance using both the linear SVM and the nonlinear rbf kernels. I ended up selecting the linear model to avoid over fitting. I also found the most descriptive C value to be 0.4.

#### Hard negative mining

I used hard nagative mining to reduce false positives. This helped a little on the testing dataset but it was unclear that this helped on the real world dataset. 

#### Result

As mentioned previously the result of the training phase was  98.5% accuracy on the test dataset. This I assumed would be good enough, but it appears the cars in the real world example were less similar to the cars from training sets. I would natually want to take examples from the real world video, but at that point I would be invalidating my results as I would could overfit on this particular video. 

### Processing

After a model was extracted from a video provided was processed and vehicles were detected. The detection process included a sliding window approach where features vectors were extracted and processed from a number of positions on each frame. Positions with high probabilities of being a car were then stored and a non maximal surpression algorithm was used to remove potential duplicate boxes on the same car. 

### Sliding window

I only actually implemented a sliding window in the x-direction. To keep the number of regions processed low I hand selected zoom y-value combinations that were more likely to produce hits. For example smaller cars are more likely to appear further up (lower y value) the image, therefore I selected a smaller zoom coupled with a proportiately high y-value (zoom: 0.7, y value 280). I hand selected seven of these zoom y-value combinations to use. 

[Sliding Window Example](https://youtu.be/RkKLiyov3zE)

### Non maximal surpression 

To achive non maximal surpression I ordered the boxes by highest probabilities and then removed boxes that overlappened based on order. This way I was left with only boxes that did not overlap much. This algorithm could have been extended to look not only at overlaps but boxes that wer close together because in a couple frames of the video two non overlapping boxes are picked for the same car. 

### Tracking

My tracking algorithm is minimal and consists of keeping a list of hits from the last frame. The required probability of boxes close to previous hits is then dramatically reduced on future frames.