# Mnist-digit-recognition

# We develop the Model using scikit-learn.k-nearest neighbors algorithm (k-NN)

The MNIST database is a dataset of handwritten digits. It has 60,000 training samples, and 10,000 test samples. Each image is represented by 28x28 pixels, each containing a value 0 - 255 with its grayscale value.

MNIST handwritten digits have been arguably the most popular dataset for machine learning research. Although the state-of-the-art learned models have long ago reached possibly the best achievable performances on this benchmark, the dataset itself remains useful to the research community, providing a simple sanity check for new methods: if it doesn't work on MNIST, it doesn't work anywhere!

We introduce n-digit variants of MNIST here. By adding more digits per data point, one can exponentially increase the number of classes for the dataset. Nonetheless, they still take advantage of the simpleness and light-weighted nature of data. These datasets provide a simple and useful toy examples for e.g. face embedding. One can furthermore draw an analogy between individual digits and e.g. face attributes. In this case, the dataset serves to provide quick insights into the embedding algorithm to be scaled up to more realistic, slow-to-train problems.

Due to potential proprietarity issues and greater flexibility, we release the code for generating the dataset from the original MNIST dataset, rather than releasing images themselves. For benchmarking purposes, we release four standard datasets which are, again, generated via code, but deterministically.
