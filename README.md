# On the Latent Space of Generative Models

Our paper is available at: https://openreview.net/forum?id=SL9w_9M-kSj


## Repository Structure

IMPORTANT NOTE: Run everything from this parent directory, relative imports used.



### Models/

The models found here are all independently/separately trainable by executing the files. After training is finished, a trained model will be stored in `TrainedModels/`. Files should be executed within the main `GenerativeLatentSpace` parent-directory. Training progress can be monitored in `Outputs/`.

Example command:
```
python Models/VAE.py --latent_dim 16
```
This will train a VAE on MNIST digits and output some latent samples.

The folder `utility/` contains some of the helper architectures that are used for e.g. improved variance and feature mappings.


### Geometry/

Most (if not all) functions involving manipulating the geometry can be found in this directory. 


#### curves

Contains Bspline and Bezier curve implementations.


#### geodesic

This contains the training of the shorter curve (not truly a geodesic in most cases). Our training scheme is as follows:

1. Start with straight line (worst case fallback). Optimize the Cubic Bspline control points (we start with 2) such that the length of the curve is minimized.
2. Start with largest learning rate for val_epochs and see if curve length has shortened:
    * If shortened: Repeat step 2 with current learning rate.
    * Else: Decrease the learning rate, reset to best curve so far, and run for val_epochs again. Repeat Step 2.
3. After Step 2 has been repeated some threshold amount of times, we add another control point to the best Cubic Bspline so far, and reset the learning rate. Repeat Step 2.
4. Termination could be maximal number of epochs or maximal number of control points.


#### metric

Here you'll find the Riemannian parts: computing the pull-back metric and Riemannian curve lengths.


### MNIST_interpolation

Creates interpolation between vectors [-1,...,-1] and [1,...,1] in latent space of a trained generator. Outputs a sequence of output images along the interpolation of both straight line and shorter curve, as well as a cross-correlation of both interpolation image sequences.

Example:
```
python MNIST_interpolation.py VAE --latent_dim 16
```

Looks for the default trained generator (with some latent dimension) named "trainedVAE_D.pth", in case it has been renamed after training, use the additional argument `--trained_gen trainedVAE_D.pth`. 



### MNIST_featureInterpolation

Creates interpolation between vectors [-1,...,-1] and [1,...,1] in latent space of a trained generator after mapping to a new output space (a feature space). Outputs a sequence of images along the interpolation of both straight line and shorter curve (in generator output space, without feature mapping), as well as a cross-correlation of both interpolation feature sequences (this time new output space with feature mapping). 

Example:
```
python MNIST_featureInterpolation.py VAE --latent_dim 16
```

Feature spaces that are currently available are: logistic regression and VGG. For VGG you require the pretrained network weights in `TrainedModels/VGG_pretrained.pth`.



### MNIST_MCimprovement

Sample N pairs of endpoints in latent space and compute the relative length improvement possible on the straight lines (by finding shorter curve). The step size is multiplied while moving in direction of the maximal eigenvector.

Example:
```
python MNIST_MCimprovement.py VAE --latent_dim 16 --N 100 --step_size 1
```


### MNIST_MCcompare

Sample N pairs of endpoints in input space and compute the relative length improvement possible on the straight lines (by finding shorter curve) FOR TWO MODELS. These two generative models are then compared with one another by looking at the respective relative length improvements. The two models can have different latent dimensions. This time both the encoder and decoder are needed, for now only BiGANs are supported in the GAN category. When trained model names are not specified, it looks for the default ones.

Example:
```
python MNIST_MCcompare.py VAE BiGAN --latent_dim1 16 --latent_dim2 16 --N 100
```

Currently assumes MNIST Digits data as test data (can be changed within file). Also only compares two models, but can be easily extended to compare multiple models (at the cost of it being more difficult to find comparable interpolations).


### generate_data

When executed will generate a directory with N samples from a certain generator:
Example:
```
python generate_data.py VAE --trained_gen trainedVAE_D.pth --latent_dim 16 --N 1000
```
The appropriate trained generator in `TrainedModels/` has to be given, along with the latent space of the model.

If no generator (currently VAE or BiGAN) is given, then it generates MNIST digits test data. Output folder is `Outputs/MNISTdigits/` for test data, and `Outputs/$generator$/` for generated data.

These folder with generated images can then be used to compute the FID score using the "pytorch-fid" module (from https://github.com/mseitzer/pytorch-fid).


### evaluation 

Contains utility functions to compare straight line and shorter curve. Currently contains two functions:

* *create_sequence*: both paths are mapped to output space (image output) and the sequence of images is plotted next to one another for visual inspection.
* *create_correlation*: the sequence of images is now further processed: we auto-correlate the sequence by taking the dot-product between all the images with eachother pairwise. The correlation matrix is then stored as output.


### dataloader

Contains the datasets that are currently usable. Currently focussed on MNIST data (28x28 grayscale images), implemented are _MNIST digits_, _Fashion MNIST_ and _EMNIST_. We also provide a class to allow custom data (however, evaluation methods are currently only implemented for 28x28 grayscale images).


## Citation
If you found our work helpful in any way, we always appreciate a citation:
```
@inproceedings{
   michelis2021on,
   title={On Linear Interpolation in the Latent Space of Deep Generative Models},
   author={Mike Yan Michelis and Quentin Becker},
   booktitle={ICLR 2021 Workshop on Geometrical and Topological Representation Learning},
   year={2021},
   url={https://openreview.net/forum?id=SL9w_9M-kSj}
}
```
