# PyNet

PyNet is a Python "package" for neural networks built from scratch. It utilizes only [NumPy](https://numpy.org/) for efficient linear algebra operations and is meant to mirror [TensorFlow](https://www.tensorflow.org/) in terms of its interface and core capabilities.

> **Disclaimer:** PyNet is not actually a Python package (hence the quotation marks). It is not published on [PyPI](https://pypi.org/), does not have any official releases, and has not been formally tested. PyNet was created as an exercise in writing neural networks from scratch and should not be used for anything serious (use a library like [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/) instead).

## Basic Usage

PyNet can be imported by:

``` python
import pynet
```

Or alternatively as:

``` python
import pynet as pn
```

It may be more useful to directly import the required classes from PyNet and its submodules:

``` python
from pynet import NeuralNetwork
from pynet.activation_funcs import Softmax
from pynet.typing import Float64, NDArray
```

The core of PyNet is the `NeuralNetwork` class and the `NDArray` data structure, which is just an alias for a [NumPy array](https://numpy.org/doc/stable/reference/generated/numpy.array.html). To fully instantiate a `NeuralNetwork`, other classes like a `Layer` and `Optimizer` are required, which in turn may utilize other classes. For example:

``` python
from pynet import NeuralNetwork
from pynet.activation_funcs import Linear, Sigmoid
from pynet.cost_funcs import MeanSquaredError
from pynet.layers import DenseLayer, DenseOutputLayer, InputLayer
from pynet.optimizers import GradientDescent
from pynet.typing import Float64, Int64, NDArray

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

For a more detailed description of what PyNet provides, see the [documentation](DOCUMENTATION.md).

## Example - MNIST Digit Classification

In [`mnist.py`](src/example/mnist.py), there is a complete example of how PyNet might be used to classify digits from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

### Data Format and Preprocessing

The original data was downloaded and preprocessed (28x28 images flattened to 784-dimensional vectors and targets were [one-hot](https://en.wikipedia.org/wiki/One-hot#Machine_learning_and_statistics) encoded) before being saved as a [compressed `.npz`](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html) file. If you wish to use the preprocessed data, download [`data.npz`](src/example/data.npz) and extract the NumPy arrays as shown in [`mnist.py`](src/example/mnist.py).

### Network Architecture

In terms of network architecture, I settled on a single hidden layer with 200 nodes with a [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function and an [elastic net](https://en.wikipedia.org/wiki/Elastic_net_regularization) regularizer. The output layer uses a [softmax](https://en.wikipedia.org/wiki/Softmax_function) activation, the same regularization, and a [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) cost function. For the learning process, I used the [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) optimization algorithm. For non-convolutional neural networks, this seems to be a fairly standard set up when it comes to [MNIST](http://yann.lecun.com/exdb/mnist/). After some trial-and-error, I was able to find optimal hyperparameters though better performance can likely be achieved by fine-tuning them more.

### Model Performance

After training the model for about 15 epochs (which took about an hour or so on my laptop), I was able to achieve parameters for the network that resulted in 97.67% accuracy for the training set and 96.57% accuracy for the test set (a cross-validation set was not used for simplicity). These parameters can be found in [`parameters.npz`](src/example/parameters.npz) and the commented-out code in [`mnist.py`](src/example/mnist.py) shows how they can be loaded into the model.

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

That being said, comparing my "library" to [TensorFlow](https://www.tensorflow.org/), which is made for large-scale deep learning applications, may be a bit unfair. PyNet's sluggish performance makes sense given that it is all in native Python (despite computations with [NumPy](https://numpy.org/)) and considering that everything was written from scratch, 97% accuracy is not terrible. A machine more powerful than my laptop would allow for a more complex architecture and with some hyperparameter tuning, PyNet could likely achieve closer to 99% accuracy.

## Limitations

As stated multiple times already, PyNet is not intended to be used as a library, even though this repository is structured like a Python package. The goal of this project was for me to deepen my understanding of neural networks by implementing one from scratch. Framing it as a package allowed me to consider the structure of the code and input validation though the main focus was on the machine learning concepts. That being said, here are some limitations with this project and concrete TODOs in order to turn PyNet into an actual package:

- **Limitation #1:** _Python is slow._ [NumPy](https://numpy.org/) was used for all linear algebra and array operations and everything is about as vectorized as possible (no explicit `for`-loops when it comes to array computations). Despite this, all of the code is in native Python and so the frequent iteration through layers during forward passes and backpropagation creates overhead. When processing 60,000 examples for 15 epochs (see the [MNIST example](README.md#example---mnist-digit-classification)), it adds up and results in an excruciatingly slow training process.

- **(Possible) Solution(s) #1:** _It is possible that more vectorization is in order, which would definitely speed up the computations._ However, this would convolute the code, demand a significant restructuring of the project, and potentially complicate how the gradients in the backpropagation process are calculated (which was already difficult for me to compute by hand). _Alternatively, rewriting the library in C++ and then calling it from Python would work._ While I do need an excuse to write more C++, this would be a difficult task given how low-level the language is and it would end up obfuscating the original goal of this project. Both of these are valid options but, at least in my opinion, not worth actually implementing as the main goal of this project has been satisfied.

- **Limitation #2:** _I (probably) wrote more bugs than tests._ While the code does run, I don't claim it to be perfect. I did catch a few bugs but no doubt there are more. To make matters worse, I did not write any tests (even though this is supposed to be a package). I wanted to focus more on the machine learning concepts in this project and not worry about writing tests for functions that mainly perform mathematical computations. That being said, I did include significant input validation for most methods which at least ensures there are no obvious errors in usage.

- **Solution #2:** _Sit down and write tests, there's no easy way around it._ If I did intend to turn this project into an actual package, this would likely be the next step in doing so. Aside from being good practice when it comes to writing code, tests would make sure that PyNet actually works as intended. But while tests would definitely help me write more robust code in the future and allow me to learn about test-driven development first-hand, it would require significant time commitment and not directly contribute to my knowledge of machine learning and neural networks.

## Acknowledgements

[NumPy](https://numpy.org/) - Used throughout PyNet for its efficient linear algebra capabilities and has made up the foundation of this project.

[Coursera Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) - I learned a lot about neural networks by taking this course and this project helped strengthen my understanding of deep learning.

[Neural-Network-Experiments](https://github.com/SebLague/Neural-Network-Experiments) - Sebastian Lague's repository about neural networks was extremely helpful in structuring this project and his video clearly explains the more difficult concepts like backpropagation.

[MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Yann Lecun's MNIST dataset provided a practical application of neural networks that is also widely known, allowing me to use PyNet in a "mini project" of sorts and benchmark its performance.

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fandrewharabor%2Fpynet&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Views&edge_flat=false)](https://hits.seeyoufarm.com)
