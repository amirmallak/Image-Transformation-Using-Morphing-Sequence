# Image-Transformation-Using-Morphing-Sequence

This project addresses Morphing Sequence Research using Transformations and Image Mappings.

The project contains a script (main.py) which walks through Researching different image to image Transformations, Morphing, and Processing. And it's divided into 3 parts:

**Part A** -- 

Creates a sequence of images morphing from one face image to another (Face images can be found in FaceImages directory).
In this part we Choose and save 12 points based on the Locations.jpeg image provided (you can choose more). Then morphing them using the functions in core_processing.py. 
In order to create a smooth video, we use number of frames large enough to create the transition (100 in our case). 
In the associated videos, we display the created morph sequence.

**Part B** --

Creates a video where the Projective Transformation calculated between pair of images performs better than an Affine one. Each of the Transformations were calculated as a Hyperplane using a 3D ambient space.

**Part C** --

This part addresses the points of choice for the Transformations affects it!
    
    * Part C - 1 --
    Shows that the number of points chosen affects the morph result.
    
    * Part C - 2** --
    Shows that the location of points chosen affects the morph result. Sparse vs Dense set of points.

The above parts research result are provided in 3 separated directories accordingly -

* Part A Directory

* Part B Directory

* Part C Directory,
    
    * Part C - 1 Directory
    
    * Part C - 2 Directory

This project works in an interactive manner by providing the user with the ability for choosing the corresponding points for determining and calculating the different transformations. Though it's possible and preferable to choose once, and load the points in future runnings (the script saves and loads the points of choice).
The script also creates and saves videos for the different image to image transformation process.
