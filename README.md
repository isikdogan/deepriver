# DeepRiver
A River Centerline Extraction Model that uses Deep Convolutional Neural Networks


## Related papers
* F. Isikdogan, A.C. Bovik, and P. Passalacqua, "Learning a River Network Extractor using an Adaptive Loss Function," *IEEE Geoscience and Remote Sensing Letters*, 2018. [[**Read at IEEExplore**]](http://ieeexplore.ieee.org/document/8319927/), [[**PDF**]](http://www.isikdogan.com/files/Isikdogan2018_deepriver.pdf)
* F. Isikdogan, A.C. Bovik, and P. Passalacqua, "Surface Water Mapping by Deep Learning," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2017. [[**Read at IEEExplore**]](http://ieeexplore.ieee.org/document/8013683/), [[**PDF**]](http://www.isikdogan.com/files/isikdogan2017_deepwatermap.pdf)

## Dependencies
* Python 3.5+
* OpenCV 3.0+
* TensorFlow 1.4.0+
* Numpy

## Running instructions
Training a model from scratch:
> python trainer.py

Running inference on a given image (extracting the centerlines from a surface water map):
> python inference.py

Example images can be found in:
> ./data