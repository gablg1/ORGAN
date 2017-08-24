# Objective-Reinforced GANs (ORGAN)

Have you ever wanted...

* to generate samples that are both diverse and interesting, like in an adversarial process (GAN)?

* to direct this generative process towards certain objectives, as in Reinforcement Learning (RL)?

* to work with discrete sequence data (text, musical notation, SMILES,...)?

Then, maybe **ORGAN** ([Objective-Reinforced Generative Adversarial Networks](https://arxiv.org/abs/1705.10843)) is for you. Our concept allows to define simple *reward functions* to bias the model and generate sequences in an adversarial fashion, improving a given objective without losing "interestingness" in the generated data.

This implementation is authored by **Gabriel L. Guimaraes** (gabriel@pagedraw.io), **Benjamin Sanchez-Lengeling** (beangoben@gmail.com), **Carlos Outeiral** (carlos@outeiral.net), **Pedro Luis Cunha Farias** (pfarias@college.harvard.edu) and **Alan Aspuru-Guzik** (alan@aspuru.com), associated to Harvard University, Department of Chemistry and Chemical Biology, at the time of release.

We thank the [previous work by the SeqGAN team](https://github.com/LantaoYu/SeqGAN). This code is inspired on SeqGAN.

If interested in the specific application of ORGANs in Chemistry, please check out [ORGANIC](https://chemrxiv.org/articles/ORGANIC_1_pdf/5309668/3).

## How to train

First make sure you have all dependencies installed by running `pip install -r requirements.txt`.

We provide a working example that can be run with `python example.py`. ORGAN can be used in 5 lines of code:

```python
from organ import ORGAN

model = ORGAN('test', 'music_metrics')             # Loads a ORGANIC with name 'test', using music metrics
model.load_training_set('../data/music_small.txt') # Loads the training set
model.set_training_program(['tonality'], [50])     # Sets the training program as 50 epochs with the tonality metric
model.load_metrics()                               # Loads all the metrics
model.train()                                      # Proceeds with the training
```

The training might take several days to run, depending on the dataset and sequence extension. For this reason, a GPU is recommended (although this model has not yet been parallelized for multiple GPUs).

