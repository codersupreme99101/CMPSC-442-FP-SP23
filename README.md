# CMPSC-442-FP-SP23

# Final Project for CMPSC 442 Spring 2023 Semester

# Topic: AI and Health (Group 1)

This document also serves as a user manual. 

# Project: Binary Classification and Efficacy Evaluation of Brain Tumors using Open Source Grayscale MRIs via Convolutional Neural Network and Visual Geometry Group - 16 layered approaches

Project Documentation (other files): https://drive.google.com/drive/folders/1IL2n0IoxxeQPzQ3ME6R2u45WYUFdpOzX?usp=sharing

(For Slides, Report, Server Related Code, etc.)

## Contributors: Arya Keni, Samantha Van Seters, Brian Truong, Vincent Purr, Yuqi He, Jiye He

## CMPSC 442 Spring 2023, Pennsylvania State University Department of Computer Science

## Group Name: Brainiacs


# Problem Statement:

This project concerns itself with binary classification of brain tumors via MRI artifacts in small grayscale MRI images across human brain samples in corss-sectional platforms, with the parallel computing capabilities of a CNN (Convolutional Neual Network) and a VGG Net (Visual Geometry Group of 16 layers). This is an important intersection of AI (Artifical Intelligence) in Health related issues, specifically that of compelx neurological disorders and complications. 

# Usage Instructions:

To utilize the code for this project, follow these steps from (1) through (10): 

(1) Clone this repository locally, or download it (through GitHub CLI, or the download .zip, or the SSH)

(2) After the clone, navigate to the directory that this repository is stored in, and open the "code" folder

(3) Open the .ipynb file with the phrase "cnn" and "vgg" in it, marking the 2 different types of Neural Net models for coding

(4) Run the cell blocks in each file block by block sequentially, from the first block onwards.

(5) For checking the result files, and plots/charts, check the "Results" folder in the downloaded directory. The presence of the "cnn" or "vgg" in the title of the file indicates the type of model the result is for, and the metric is present in the file name as well (and in the plot title, with the model). 

(6) For the pip installs, refer to the "!pip" statements in the.ipynb file, which can be run within the file for dependencies. 

(7) Make sure you are logged in to your google account for all the following operations, and allow access to your google drive for file access of data and results. (The system will auto-prompt for this).

(8) Ensure that you are connected to a running instance by checking the right hand corner that mentions "connected" in green. This should automatically occur, and if not, it will in 3-4 minutes after a page refresh. 

(9) Click "Runtime" and "Computing" on the top menu bar to check that the instance is connected to any CPU at least (or a GPU/TPU works better, if there is personal access to it). 

(10) For any bugs/errors (which there should not be any), double check for any red underlines/red blocks in the code blocks, one-by-one, and edit to fix (since it may be typo, by user edit). 

# Design:

The design is that of 2 separate systems that process the image data of the MRIs as follows:

CNN: 

Filtered image by filter algorithms followed by repeated convolution, pooling, and activation in 3 layers of sets. Feature Maps are applied and pooled in 2 more layers, and then vecotized into the final labels.

VGG Net (16):

Convolution, Fully Connected Layering, Activation, Max Pooling and Layering in Softmax, with layers in number as follows: 5, 1, 2, 1 (repeated till 16 in space) (for a sum of 16)

Further explanation of high-level image diagrams can be found in the presentation slides of this repository. 

# Implementation:


Using Jupyter Notebooks and Google Colab, for enhanced GPU capabilities for shorter training times. Fully written in python, with the libraries: Keras, Tensorflow, Matplotlib, Numpy

Implemented in code blocks, and file driectory of dataset accessed from the googl drive cloud. 

Analysis of plots and charts done in same singular code file as training and validation, along with initialization of libraries. 


# Algorithms Used:

Using the Keras and Tensorflow libraries has provided custom and direct implementations of the following algorithms:

For NN (Neural NEtwork) Layers: 

Linear Activation, Softmax function, Fully connected algorithm, Convolution as an integral function, Max Pooling Algorithm, Pooling Algorithm. 

For Post-Processing: 

K-Nearest Neighbors, Denoising Images Algorithm Libraries, Principal Component Analysis (PCA)

For Statistical Analysis: 

Recall Algorithm, F1 Algorithm.

(Note that basic algorithms and methods, such as Mean Squared Error, Confidence, and Accuracy Computations are not listed)

# Ethical and Moral Considerations:

From the context of morality, there are no issues with infringement or immoral activities since this application is deployed in the medical setting, and will have the ethical rules and considerations of a medical institution in place, which prevent issues of this sort. 

There may be the ethical and moral consideration of reliance on such a system, and limitations and boundaries of this system being towards the data it is trained on it being strictly labelled with 2 versions, which is not sufficient for any advanced diagnossis. But the convenience and general accuracy of such a system may prove to be harmful if disregarded as a classification regime, and nothing more. 

More data of differing MRI artifacts can be trained upon this model set, but the issue here is that for the intricacies of human brain related tissue mapping, even in 2D across all disease vectors as labels is a very long term project, that may intertwine more with the ethics of utilization of such systems. 

# Goals, Environment and Adaptations (Short Version, refer to GEA.pdf for details):

## Goal:

The system aims to identify the existence of a tumor or not, based on the specifications given in
a unitary MRI scan for a sample human model (regardless of any other factor, and only
considering that of the brain tissue structure and physical features). The MRI scan being
naturally grayscale, and giving a detailed reconstruction of neurolateral signatures is useful to
exploit in a network that learns such features.

## Environment:

The MRI scans for thousands of input data (split to a train and test set in an acceptable ratio) that
are open source from a biological institute for neurological diseases are used as the key driving
factor in the binary decision making process. There may be other nuances within the scans
indicating some sub-form of brain tumors, but the key idea here is to identify the possibility of an
abnormality, and its leading implications in such a manner.

Thus, the machine learning algorithm will use many layers of condensed data to extract features
that are alarmingly non-uniform given the general structure. Interestingly, this is also done in
context of the relative grayscale distributions, and what that may mean with the lobes/regions of
the brain. This lends itself to the physics behind water polarization in areas of the brain, and how
its effect is limited/enhanced depending on tissue growth and deformities, clearly correlated to
the grayscale pixels in a small region of interest.

The above images serve as a reference to a human-distinguishable set of binary classification for
tumor existence, to give an idea of what the neural net learns to look for on an exponentiated
level of complication.

Another important consideration is that of the stakeholders involved in the utilization of this
system. Depending on the advancement of this system in classification ability per vector label
size, the use of this can span from neurologists and neurosurgeons that are relatively
inexperienced, to the experts in the field as well. This is because it can be used reliably (ideally)
for tumor identification by a single scan upon sufficient training, which will help millions in
terms of complicated brain related diseases and injuries as a course of action can be employed
more effectively. The downside to this might be that too much reliance on such a system may
destroy the ability to critically think and provide a second opinion on such predictions, or even
prevent the goal of research in adapting the system to other labels or parameters, something that
is currently done in a more conventional biochemical manner that doesn't involve Neural Nets
(and neither is this proposed system geared towards that).

## Adaptation:

Typically, a convolutional or image processing based learning network learns to identify the
features in a supervised manner (such as its related sub-classes of VGG net or DGCNN net as
well, that use graph theory and/or deep layering architectures to enhance certain aspects of their
feature extraction), where the labels for tumor existence or not are given for a certain percentage
of input images, and the system learns to find common features in areas of grayscale values and
image structure by artifact, to apply to “unlabelled” testable data, and then use that as a range of
metric possibilities to evaluate how well it performed in that regard (recall quality, accuracy,
confidence in the binary prediction, and more.

Neighboring pixels and their relative distributions are a key computational aspect to CNNs
(Convolutional Neural networks or Nets) and their family of nets, where filters of these kernel
samples, and the connectivity between them to condense and sample the image space is
important in terms of clustering/grouping of certain attributes that the intermediate layers deem
“desirable” during the learning epochs.

# Social Implications:

This will benefit the medical community and communities of large populations in general, due to the effect this technology can have in potentially preventing and providing (a course of action) brain tumors in a general sense, for a broader range of physiological differences. This will speed up treatment timelines, and will provide a very unbiased and specifci second opinion at the least to any medical professional. 

A Utopian outlook on this implication may mean the eventual learning and eradication plan generation that this AI can perform after sufficient training. This can directly translate to a potential cure for most of the neurological diseases and disorders stemming from physiological impairments, not just tumors. After the system trains on billions on parameters, image types, and classification specifics, it can be an unmatched model for medical diagnosis n modern neuroscience. 

On the other hand, this system (in a dystopian setting), though accurate within its on accord, can produce oversaturation and inept decision making skills at the hands of global professionals, leading to a dystopian reality where the ability for the future to accurately decide upon and eradicate/solve neurological conditions will diminish, or cause harm. Alternatively, the system may never learn any more due to its exceeding dependency, causing a halt in data science based biomedical applications in brain related diagnostics and imaging. 
