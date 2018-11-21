# Noise2Noise Binning

*Axel Ekman, Jian-Hua Chen, Venera Weinhardt, Gerry McDermott, Mark A. Le Gros, Carolyn Larabell*

As Presented by: [Axel Ekman](mailto:axel.ekman@iki.fi)
at
[CAMERA Workshop](http://microct.lbl.gov/cameratomo2018)
October 31 - November 2, 2018

## Intro
In terms of signal processing, the optimal digital filter to remove the high-frequency portion of the image is the sinc filter. When decimation is doen by an integer factor, area-averaging is usually very close to optimal and produces usually not much aliasing. In this case, downsampling by a factor of 2 can be expressed in the from

![downsampling](images/latex_downsample.png)

Ideal filters like this are unbiassed and do not take into account any priors that may be suitable for the image. The basic idea of this method is that we can construct separate signals from the data and train a CNN to do the downsampling.

Recent work of Lehtinen et al. show that instead of needing true signal, CNNs can be trained using just noisy images by minimizing some distance (loss function) L.

![n2nfilter](images/latex_n2nfilter.png)


Based on
[Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189)

```
Lehtinen, Jaakko, et al. “Noise2Noise: Learning Image Restoration without Clean Data.” <em>Proceedings of the 35th International Conference on Machine Learning, PMLR 80:2965-2974, 2018</em>.
```

Now within the sampling rate of the output image, we can view all pixels corresponding to the same binned pixel as separate observations of the downsampled image. This provides information to optimize some parametrized filter such that we can use the result of Lehtinen et al. to train a CNN sownsampler.

![n2nfilter](images/latex_n2nbin.png)

where X1 and X2 are two uncorrelated data samples from the high-resolution image. This can be e.g. done by dividing each downsampled pixel into two diagonal regions (the fact that the center-of-mass is the same should take care of some sub-pixel artifacts). One could also choose random samples of the square to construct several permutations of the same image. In practice this made little difference in the results.

![Schematic](images/schematic.png)


## Examples

### Color images

Below we show the result for 'monarch' in SET14 with artififcial Gaussian  noise (sigma = 30) compared to the mean-binned image. The reference methods (TV, NLM, BM3D) show the corresponding result by minimzing the true loss function using oracle information of the the reference image.

![Schematic](images/monarch_example.png)

### Tomography

Example of a SXT reconstruction of a Human B-cell reconstructed with FBP (Ram-Lak). In this example, the net was trained simultainiously on all projection images.

|Binned projections | CNN binned projections |
| ---|---|
|![FBP noisy](images/fbp_stackm_noisy.png)|![FBP filtered](images/fbp_stackcnn_noisy.png)|








