#!/usr/bin/env python
# coding: utf-8

# # Data Versioning Done Right.
# 
# ---------------------------------------------------------
# 
# Reproducibility is one of the most important problems in mid- to large-scale ML projects. As the project grows, it accumulates dependencies, legacy and unclear use cases and keeping all that mess organized sometimes requires big changes to development practices and infrastructure. Reproducibility is a multifaceted issue with many aspects to it, but today I want to talk particularly about data versioning.
# 
# Data versioning is a practice/tool which allows you to keep your datasets and training code in sync. As the project grows so does the dataset. The training data is often composed of multiple open-source and private dataset which are stored in different formats, serve different purposes and require revision every now and then. This makes it really easy to forget which data a particular version of your model was trained on. When you add preprocessing into the mix, things get even more complicated. Data preprocessing is often a time-consuming calculation, and I've seen how people (and me as well) follow this anti-pattern of writing a standalone data preprocessing script, which dumps results into some folder and then use that folder as a "dataset" for the model training. It is pretty bad for reproducibility on itself but it can also lead to even bigger problems later on.
# 
# > **Example:** Suppose your dataset consists of photos of people, but you need only their faces. So you run your photos through a face detector, crop faces, resize them to a desired size and save to a directory. Now imagine that you changed parameters of the face detector to make wider face crops. The image size stays the same and two folders of generated crops (tight and wide) will be indistinguishable for a training code on their own, so technically you can use wrong data. To prevent that you need to manually check the version of the data used, and what is done manually is prone to mistakes. As your preprocessing pipeline becomes bigger it will require more and more manual checking. This gets even worse when multiple people work on the project from different machines, as they may have different folder structure and just pass wrong paths to the training code. If two people conduct independent experiments and one of them unintentionally uses the wrong data, it will cause conflicting results which are hard to debug and may lead research into the wrong direction and cost a lot of time.
# 
# Besides the obvious problems with this approach there's a hidden one. When a team eventually decides to implement data versioning into their project they can easily confuse these preprocessed folders with the actual dataset and try to apply data versioning to them instead. Conceptually there's nothing wrong with it: that folder is data which can change, the changes must be tracked, and there's a tool for tracking changes in the data. The problem, however, hides in a way data versioning tools perform tracking. So let's talk about that a bit.
# 

# # Data Versioning
# 
# There are several such solutions for data versioning out there, which share a lot of functionality. I myself mostly have to deal with [DVC](https://dvc.org/) so I will be using it as an example.
# 
# 
# Data versioning systems like DVC provide two main features along other bells and whistles:
# 
# 1. Detect changes in a data folder using some sort of hashing and record these changes into a register.
# 2. Provide some storage management to sync these registers between different providers: local storage, S3, GCS, databases etc.
# 
# But because datasets are mostly stored as binaries, there's no way for a DVC to calculate meaningful diffs between files. Everything it can do is to compare hashes and create a copy of a file if hashes don't match. All those copies are stored in a remote data repo separately from the code (in a S3 bucket, for example) and are linked to your project via a registry file added to your normal Git repo. You can probably guess where this is going - we can only have so many versions of our dataset before our storage blows up. This is crucial when we try to manage automatically generated data using DVC. Any tiny change to the data processing code completely messes up hashes so DVC creates a full copy of the data on the remote storage. 
# 
# > **Example:** On one of my projects a dataset contained 270GB of photos of people. Each change to a preprocessing codes will chew away 270B at a time and after only 10 changes you'll get a 3TB storage bill for what can be described by a handful of bytes in Git. And it is hard to impress anybody with a 270GB dataset nowadays, when models like Stable Diffusion are trained on 1000x  larger data.
# 
# This poses a dilemma: 
# - If your dataset experiences frequent changes, this suggests using data versioning, which is very storage inefficient. 
# - If your dataset is stable, then you don't need data versioning - there will be just a single commit.
# 
# But the truth is that in cases like that you actually don't need data versioning. You need caching.

# # Data Caching
# 
# The main reason to create a folder with preprocessed data is to speed up your work: you memorize the outputs of your preprocessing code, so when you need them again, you can just read them from disk instead of recomputing. This is what caching does. But the interaction with the cache is different, and this is what causes problems. Here's an example of the correct usage of caching:

# In[1]:


@cache
def process_data(sample):
    # some transformation of the sample
    # ...

x = process_data(69)  # first use - results are computed
use_data(x)

# other stuff is done here
# ... 

x = process_data(69)  # second use - results are read from cache
use_data(x)


# Notice how data preprocessing is called both times. This ensures that `process_data` is always in sync with `use_data` - they are called together despite the state of the cache. But this is how caching is done in our folder-with-files case:

# In[2]:


x = process_data(69) 
save_to_cache_folder(x)

# a possible break in execution
# ... 

x = load_from_cache_folder()
use_data(x)  # first use - results are read from cache

x = load_from_cache_folder()
use_data(x)  # second use - results are read from cache


# Here `process_data` and `use_data` are not in sync and all sorts of bad things can happen. If we'd change our code to the correct one - with redundant calls to the `process_data` function, then we would achieve two things:
# 
# 1. We would ensure that our preprocessing code does not diverge from the training code, because it gets called every single time and not only at the start of the project
# 2. Any change to the preprocessed data would be reflected in the preprocessing code, which is managed by Git and takes no storage compared to a complete new dump of data created by DVC.
# 
# Having that the only data changes which would go to a DVC repo would be some manual changes - the only changes which cannot be stored in any other way. This way our DVC repo would have minimal size without compromises to traceability. 

# # Cache Design
# 
# Unfortunately, DVC does not provide tooling necessary for such data caching. So let's think of potential use cases and design our cache more accurately
# 
# > **Requirement #1:** Cache must be persistent between multiple runs of the training pipeline on the same working machine, restarts of the working machine and even multiple working machines.
# 
# The last one is needed so the different people can reuse the same dataset with costly recomputation. These requirements suggest a disk-based cache. Its read times are on par with plain files; it is essentially a folder with files but smarter, so it lives as long as storage does and it can be synced between multiple machines using various tools.
# 
# > **Requirement #2:** Any change of the cacheable function which changes its outputs must lead to a recomputation of the cached value.
# 
# This requirement is trivially satisfied in memory-based caches, because cacheable functions usually don't change in runtime. But for a disk-based cache there's no such guarantees. More so, we build this cache exactly to catch code changes and bind them to data. This validation mechanism needs to compare a value stored in the cache to a result recomputed during every run. This requires access to raw function arguments. It can be a problem. Arguments to a cacheable function can be tensors or arrays which take up a lot of memory. So here follows the next requirement: 
# 
# > **Requirement #3:** Cache keys must have minimal memory footprint.
# 
# In other circumstances we could use hashes, but not in our case. We can solve this issue, however. Because DVC takes care of raw data, we can pass paths to files with raw data to our processing functions instead of actual data, so the function signatures can be stored efficiently. 
# 
# > **Requirement #4:** Cache must have multiple levels for heavy data blobs and lightweight metadata records.
# 
# A data record may not be a unitary blob of data but have multiple fields. We can roughly divide all the extracted data into four groups:
# 
# |                    | **Compute - Slow** | **Compute - Fast** |
# |:-------------------|:-------------------|:-------------------|
# | **Memory - Low**   | Hard to compute, easy to store. <br> Examples: *class indices, <br> embeddings, descriptors etc.* <br><br> Cached. | Info extracted from function arguments. <br> Examples: *IDs, tags, labels extracted <br> from file paths.* <br><br> Not cached |  
# | **Memory - High**  | Hard to compute, hard to store. <br> Examples: *segmentation masks, <br> features, generated images etc.* <br><br> Cached. | Simple ops on raw data. <br> Examples: *image normalization, <br> resize, flip etc.* <br><br> Not cached |
# 
# The first group consists of some lightweight objects which are extracted from the raw data using ML models or other slow algorithms. This data is typically used for sampling and/or filtering further in the data pipeline, so we need to be able to prefetch such data from the cache into memory. On the other hand the second group includes data which cannot be fit into memory in large numbers and must be loaded lazily. The data record may consist of both types of data so our cache should support both as well. 
# 
# These are four main requirements to our data preprocessing cache. There are plenty of Python tools for caching but it seems like they fall short in one way or another. I haven't been able to find disk caching tools with appropriate invalidation strategies - most of them use TTL of some kind, which is not what we want, as our cache can potentially live for months. Also two-level caching with lazy reads is kinda a specific requirement, so I haven't been able to find anything like that as well. For this reason we'll resort to writing our own solution. We will definitely face fun engineering challenges there. But that's it for today.
# 
# Until next time.
