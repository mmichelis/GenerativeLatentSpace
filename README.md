# The Latent Space of Generative Models



## Repository Structure

IMPORTANT NOTE: Run everything from this parent directory, relative imports used.



### Models

The models found here are all independently/separately trainable by executing the files. After training is finished, a trained model will be stored in `TrainedModels/`. Files should be called within the main `GenerativeLatentSpace` parent-directory. Training progress can be monitored in `Outputs/`.

Example command:
```
python Models/VAE.py --latent_dim 16
```
This will train a VAE on MNIST digits and output some latent samples as well.


### MNIST_interpolation




### generate_data

When executed will generate a directory with N samples from a certain generator:
Example:
```
python generate_data VAE --trained_gen trainedVAE_decoder.pth --latent_dim 16 --N 1000
```
The appropriate trained generator in `TrainedModels/` has to be given, along with the latent space of the model.

If no generator (currently VAE or BiGAN) is given, then it generates MNIST digits test data. Output folder is `Outputs/MNISTdigits/` for test data, and `Outputs/$generator$/` for generated data.

These folder with generated images can then be used to compute the FID score using the "pytorch-fid" module (from https://github.com/mseitzer/pytorch-fid).
