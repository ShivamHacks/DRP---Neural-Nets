A digit classifier built from scratch.

Algorithm:
1. From training images, create "Image Stacks" which average all
   all of training images for each label. See imagestacks images.
2. For test set, calculate Euclidian distance between matrix of
   training image and each imagestack. Which ever imagestack has
   least distance from test image, test image is labeled with the
   label of that image stack.

Made at HopHacks2017