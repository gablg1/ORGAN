# Objective-Reinforced GANs (ORGAN)

* Want the diversity and interestingness that you get with samples from an adversarial process (GAN)?

* Want the directed focus you can give algorithms with Reinforcement Learning? (RL)

* Working with discrete sequence data (text, molecular SMILES, abc musical notation ,etc.)?

Then **ORGAN** if for you, define simple reward functions and alternate between adversarial and reinforced training.

Based on work from [](arxiv link here)

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
*

## Make your own experiment


## Dockerfile
