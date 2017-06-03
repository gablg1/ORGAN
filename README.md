# Objective-Reinforced GANs (ORGAN)

* Want the diversity and interestingness that you get with samples from an adversarial process (GAN)?

* Want the directed focus you can give algorithms with Reinforcement Learning? (RL)

* Working with discrete sequence data (text, molecular SMILES, abc musical notation ,etc.)?

Then **ORGAN** is for you, define simple objective functions to bias the model and generate sequences in an adversarial fashion. ORGAN improves on the given objective without losing interestingness in the generated data.

Based on work from [https://arxiv.org/abs/1705.10843](https://arxiv.org/abs/1705.10843)

## How to train

In order to train the model, cd into `model` and run

```python train_ogan.py exp.json```

where **exp.json** is a experiment configuration file.

A GPU is recommended since it can take several days to run, depending on dataset and sequence extension, algorithm is not parallelized for multiple GPUs.

## How to sample


## Requirements to run

* Tensorflow 1.0
* Python 2 or 3
* rdkit for molecular purposes
* More in requirements.txt (install with `pip install -r requirements.txt`)

## Make your own experiment

Coming soon

## Dockerfile

Coming soon

Note: this code is based on [previous work by the SeqGAN team](https://github.com/LantaoYu/SeqGAN). Many thanks to SeqGAN.
