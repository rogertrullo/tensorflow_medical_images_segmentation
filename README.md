# tensorflow_medical_images_segmentation
Here I post a code for doing segmentation in medical images using tensorflow.
First you would need to have the python packages h5py, SimpleITK and of course TensorFlow.

To use it, first I assume that you have niftii files (.nii.gz). Also, I assume that all the training images are in a folder that contains the training subjects as folders in it. Each training subject then will be a folder, and the name of this folder should be the name of the CT image. The groundtruth should be in the same folder and it should be called GT.nii.gz. It should be an image with values for each voxel rangon from 0 to num_classes-1.

Data
|
|--sub1/
    |--sub1.nii.gz
    |--GT.nii.gz
|--sub2/
    |--sub2.nii.gz
    |--GT.nii.gz
|
|...

The name can be different as long as the CT file and folder are the same.

The first hing that you'd want to do is to convert the CT and it's groundtruth data to h5 format. This is done by the generate_2d_h5.py script. To run it, you should type:

python generate_2d_h5.py --src /path/to/patients --dst /path/to/save/h5/files

Now this will generate a folder that includes several h5 files that contain the training data (input ct slices and it's corresponding labels)
