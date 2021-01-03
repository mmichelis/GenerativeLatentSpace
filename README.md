# The Latent Space of Generative Models



## Repository Structure

IMPORTANT NOTE: Run everything from this parent directory, relative imports used.

### generate_data

When executed will generate a directory with N samples from a certain generator. This can then be used to compute the FID score using the "pytorch-fid" module (from https://github.com/mseitzer/pytorch-fid)


### Models

The models found here are all independently/separately trainable by executing the files. After training is finished, a trained model will be stored in `TrainedModels/`. Files should be called within the main `GenerativeLatentSpace` parent-directory. Training progress can be monitored in `Outputs/`. 

Example command:
```
python Models/VAE.py
```
This will train a VAE on MNIST digits and output some latent samples as well.


