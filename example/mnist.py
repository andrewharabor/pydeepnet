from __future__ import annotations

from pathlib import Path

import numpy as np

from pynet import Float64, Int64, NDArray, NeuralNetwork
from pynet.activation_funcs import ReLU, Softmax
from pynet.cost_funcs import CrossEntropy
from pynet.error_metrics import PercentCorrect
from pynet.layers import DenseLayer, DenseOutputLayer, InputLayer
from pynet.normalizers import ZScore
from pynet.optimizers import Adam
from pynet.regularizers import ElasticNet

# Path to current working directory
CWD: Path = Path(__file__).resolve().parent

# Hyperparameters
HIDDEN_LAYER_SIZE: Int64 = Int64(200)
RELU_LEAK: Float64 = Float64(0.01)
REG_PENALTY: Float64 = Float64(0.0001)
LEARNING_RATE: Float64 = Float64(0.001)
DECAY_RATE1: Float64 = Float64(0.9)
DECAY_RATE2: Float64 = Float64(0.999)
EPOCHS: Int64 = Int64(15)
BATCH_SIZE: Int64 = Int64(32)

# Load MNIST data
with np.load(CWD / "data.npz") as data:
    train_inputs: NDArray = data["train_inputs"]
    train_targets: NDArray = data["train_targets"]
    test_inputs: NDArray = data["test_inputs"]
    test_targets: NDArray = data["test_targets"]

# Create, train, and evaluate neural network
network: NeuralNetwork = NeuralNetwork(
    InputLayer(Int64(train_inputs.shape[1]), ZScore()),
    [
        DenseLayer(HIDDEN_LAYER_SIZE, ReLU(RELU_LEAK), ElasticNet(REG_PENALTY, REG_PENALTY)),
    ],
    DenseOutputLayer(Int64(train_targets.shape[1]), Softmax(), ElasticNet(REG_PENALTY, REG_PENALTY), CrossEntropy()),
    Adam(LEARNING_RATE, DECAY_RATE1, DECAY_RATE2),
    PercentCorrect()
)

network.train(train_inputs, train_targets, EPOCHS, BATCH_SIZE)

# Load previously trained parameters instead of training
# with np.load(BASE_PATH / "parameters.npz") as parameters:
#     hidden_layer_weights: NDArray = parameters["hidden_layer_weights"]
#     hidden_layer_biases: NDArray = parameters["hidden_layer_biases"]
#     output_layer_weights: NDArray = parameters["output_layer_weights"]
#     output_layer_biases: NDArray = parameters["output_layer_biases"]
# network.set_parameters([(hidden_layer_weights, hidden_layer_biases)], (output_layer_weights, output_layer_biases))
# if network.input_layer.normalizer is not None:
#     network.input_layer.normalizer.fit(train_inputs)  # normalizer is only fit to inputs in the `NeuralNetwork.train()` method

train_predictions: NDArray = network.predict(train_inputs)
test_predictions: NDArray = network.predict(test_inputs)
print(f"Train Data Accuracy: {(100 * network.evaluate(train_targets, train_predictions)):.2f}%")
print(f"Test Data Accuracy: {(100 * network.evaluate(test_targets, test_predictions)):.2f}%")
