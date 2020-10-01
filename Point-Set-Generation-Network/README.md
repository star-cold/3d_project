## This repo is out-dated, and will no longer be developed. Thank you for your interest. If you are still interested, please refer to other modern implementation.

# Point-Set-Generation-Network

It is a pytorch implementation of [A Point Set Generation Network for 3D Object Reconstruction from a Single Image ][1]

Dataset can be download from here: [dataset](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html). 

## abstract 
Generation of 3D data by deep neural network has been attracting increasing attention in the research community. The majority of extant works resort to regular
representations such as volumetric grids or collection of images; however, these representations obscure the natural invariance of 3D shapes under geometric transformations, and also suffer from a number of other issues. In this paper we address the problem of 3D reconstruction from a single image, generating a straight-forward form of output – point cloud coordinates. Along with this problem arises a unique and interesting issue, that the groundtruth shape for an input image may be ambiguous. Driven by this unorthodox output form and the inherent ambiguity in groundtruth, we design architecture, loss function and learning paradigm that are novel and effective. Our final solution is a conditional shape sampler, capable of predicting multiple plausible 3D point clouds from an input image. In experiments not only can our system outperform state-ofthe-art methods on single image based 3d reconstruction benchmarks; but it also shows strong performance for 3d shape completion and promising ability in making multiple plausible predictions.

## What is implement in this tensorflow project is

- point set generation network

- squeezenet1_1


the structure is something like this:
![model](https://user-images.githubusercontent.com/10870023/31383934-ba9a5a70-ad71-11e7-8826-018c3d6a5216.png)

[1]: http://ai.stanford.edu/~haosu/papers/SI2PC_arxiv_submit.pdf




