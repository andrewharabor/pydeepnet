# PyDeepNet

PyDeepNet is a Python "package" for neural networks built from scratch. It utilizes only [NumPy](https://numpy.org/) for efficient linear algebra operations and is meant to mirror [TensorFlow](https://www.tensorflow.org/) in terms of its interface and core capabilities.

> **Disclaimer:** PyDeepNet is not actually a Python package (hence the quotation marks). It is not published on [PyPI](https://pypi.org/), lacks documentation, and does not have any official releases. PyDeepNet was created as an exercise in neural networks and should not be used for anything serious (use a library like [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/) instead).

## Basic Usage

PyDeepNet can be imported by:

``` python
import pydeepnet
```

Or alternatively as:

``` python
import pydeepnet as pdn
```

It may be more useful to directly import the required classes from PyDeepNet and its submodules:

``` python
from pydeepnet import NeuralNetwork
from pydeepnet.activation_funcs import Softmax
from pydeepnet.typing import Float64, NDArray
```

The core of PyDeepNet is the `NeuralNetwork` class and the `NDArray` data structure, which is just an alias for a [NumPy array](https://numpy.org/doc/stable/reference/generated/numpy.array.html). To fully instantiate a `NeuralNetwork`, other classes like a `Layer` and `Optimizer` are required, which in turn may utilize other classes. For example:

``` python
from pydeepnet import NeuralNetwork
from pydeepnet.activation_funcs import Linear, Sigmoid
from pydeepnet.cost_funcs import MeanSquaredError
from pydeepnet.layers import DenseLayer, DenseOutputLayer, InputLayer
from pydeepnet.optimizers import GradientDescent
from pydeepnet.typing import Float64, Int64, NDArray

network: NeuralNetwork = NeuralNetwork(
    InputLayer(Int64(10), None),
    [
        DenseLayer(Int64(15), Sigmoid(), None),
        DenseLayer(Int64(5), Sigmoid(), None),
    ],
    DenseOutputLayer(Int64(1), Linear(), None, MeanSquaredError()),
    GradientDescent(Float64(0.1)),
    None
)
```

This creates a small neural network that sequentially connects dense layers. The `InputLayer` contains 10 nodes and does not use a `Normalizer`. The `DenseLayer`s have 15 and 5 nodes, use `Sigmoid` activation functions, and do not use a `Regularizer`. The `OutputLayer` contains 1 node, a `Linear` activation function, no `Regularizer`, and a `MeanSquaredError` cost function. The network is trained via basic `GradientDescent` with a learning rate of 0.1. An `ErrorMetric` is not supplied.

As long as the training and testing data is of the correct size (10 inputs per example to match the `InputLayer` and 1 target per example to match the `DenseOutputLayer`), the model can be trained as follows:

``` python
network.train(train_inputs, train_targets, Int64(5), Int64(32))
predictions: NDArray = network.predict(test_inputs)
```

This trains the network for 5 epochs and a mini batch size of 32. Afterwards, the network's predictions are stored in an array.

## Example - MNIST Digit Classification

In [`mnist.py`](example/mnist.py), there is a complete example of how PyDeepNet might be used to classify digits from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

### Data Format and Preprocessing

The original data was downloaded and preprocessed (28x28 images flattened to 784-dimensional vectors and targets were [one-hot](https://en.wikipedia.org/wiki/One-hot#Machine_learning_and_statistics) encoded) before being saved as a [compressed `.npz`](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html) file. If you wish to use the preprocessed data, download [`data.npz`](example/data.npz) and extract the NumPy arrays as shown in [`mnist.py`](example/mnist.py).

### Network Architecture

In terms of network architecture, the example uses a single hidden layer with 200 nodes, a [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function, and an [elastic net](https://en.wikipedia.org/wiki/Elastic_net_regularization) regularizer. The output layer uses a [softmax](https://en.wikipedia.org/wiki/Softmax_function) activation, the same regularization, and a [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) cost function. Rather than basic gradient descent, the [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) optimization algorithm is used for the learning process. For non-convolutional neural networks, this seems to be a fairly standard set up when it comes to [MNIST](http://yann.lecun.com/exdb/mnist/).

### Model Performance

After training the model for about 15 epochs (which took about an hour or so on my laptop), I was able to achieve parameters for the network that resulted in 97.67% accuracy for the training set and 96.57% accuracy for the test set (a cross-validation set was not used for simplicity). These parameters can be found in [`parameters.npz`](example/parameters.npz) and the commented-out code in [`mnist.py`](example/mnist.py) shows how they can be loaded into the model.

Unfortunately, 97% accuracy pales in comparison to the most optimized models, which score upwards of 99.5% in accuracy. Even the short script on the [TensorFlow home page](https://www.tensorflow.org/) can achieve similar if not slightly better results after only a few minutes of training:

``` python
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

That being said, using [TensorFlow](https://www.tensorflow.org/) as a baseline may be a bit unfair as it is optimized for large-scale deep learning applications. Considering that PyDeepNet was written from scratch and is not particularly efficient, 97% accuracy is not terrible. Its sluggish performance is likely due to the fact that it is all in native Python, despite linear algebra computations with [NumPy](https://numpy.org/). With a computer more powerful than my laptop, a more complex architecture could be used, and with some hyperparameter tuning, PyDeepNet could likely achieve closer to 99% accuracy.

## Limitations

As stated multiple times already, PyDeepNet is not intended to be used as a library, even though this repository is structured like a Python package. The goal of this project was for me to deepen my understanding of neural networks by implementing one from scratch. Framing it as a package allowed me to consider the structure of the code and write tests though the main focus was on the machine learning concepts.

The largest bottleneck in PyDeepNet's performance is that Python is slow. While [NumPy](https://numpy.org/) was used extensively, the code is not optimized for efficiency and so the constant iteration through layers (during forward passes and backpropagation) creates significant overhead. When processing 60,000 images for 15 epochs (see the [MNIST example](README.md#example---mnist-digit-classification)), it adds up and results in an excruciatingly slow training process.

It is possible that more vectorization is needed, which would definitely speed up the computations. However, this comes with the risk of convoluting the code and would demand a large restructuring of the project. Alternatively, (re)writing PyDeepNet in C++ would give the needed performance boost. Though while this would be extremely helpful in terms of learning C++, the original goal of this project has already been fulfilled and so there's no incentive to rewrite the library simply to optimize its performance.

## Acknowledgements

[NumPy](https://numpy.org/) - Used throughout PyDeepNet for its efficient linear algebra capabilities and has made up the foundation of this project.

[pytest](https://docs.pytest.org/en/8.2.x/) - Used for writing table-driven tests to ensure that the core of PyDeepNet functions as expected.

[Coursera Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) - I learned a lot about neural networks by taking this course and this project helped strengthen my understanding of deep learning.

[Neural-Network-Experiments](https://github.com/SebLague/Neural-Network-Experiments) - Sebastian Lague's repository about neural networks was extremely helpful in structuring this project and his video clearly explains the more difficult concepts like backpropagation.

[MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Yann Lecun's MNIST dataset provided a practical application of neural networks that is also widely known, allowing me to use PyDeepNet in a "mini project" of sorts and benchmark its performance.
