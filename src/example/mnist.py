from __future__ import annotations

from tarfile import TarFile, open

import numpy as np

from pynet import NeuralNetwork
from pynet.activation_funcs import ReLU, Softmax
from pynet.cost_funcs import CrossEntropy
from pynet.error_metrics import PercentCorrect
from pynet.layers import DenseLayer, DenseOutputLayer, InputLayer
from pynet.normalizers import ZScore
from pynet.optimizers import Adam
from pynet.regularizers import ElasticNet
from pynet.typing import Float64, Int64, NDArray

# Paths
BASE_PATH: str = "src/example"
DATA_PATH: str = f"{BASE_PATH}/data"
PARAMETERS_PATH: str = f"{BASE_PATH}/parameters"

# Hyperparameters
HIDDEN_LAYER_SIZE: Int64 = Int64(200)
RELU_LEAK: Float64 = Float64(0.01)
REG_PENALTY: Float64 = Float64(5e-4)
LEARNING_RATE: Float64 = Float64(0.02)
DECAY_RATE1: Float64 = Float64(0.9)
DECAY_RATE2: Float64 = Float64(0.999)
EPOCHS: Int64 = Int64(15)
BATCH_SIZE: Int64 = Int64(32)

# Load MNIST data
data_file: TarFile
with open(f"{BASE_PATH}/data.tar.gz", "r:gz") as data_file:
    data_file.extractall(BASE_PATH)
train_inputs: NDArray = np.load(f"{DATA_PATH}/train_inputs.npy")
train_targets: NDArray = np.load(f"{DATA_PATH}/train_targets.npy")
test_inputs: NDArray = np.load(f"{DATA_PATH}/test_inputs.npy")
test_targets: NDArray = np.load(f"{DATA_PATH}/test_targets.npy")

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

# hidden_layer_weights: NDArray = np.load(f"{PARAMETERS_PATH}/hidden_layer_weights.npy")
# hidden_layer_biases: NDArray = np.load(f"{PARAMETERS_PATH}/hidden_layer_biases.npy")
# output_layer_weights: NDArray = np.load(f"{PARAMETERS_PATH}/output_layer_weights.npy")
# output_layer_biases: NDArray = np.load(f"{PARAMETERS_PATH}/output_layer_biases.npy")
# network.load_parameters([(hidden_layer_weights, hidden_layer_biases)], (output_layer_weights, output_layer_biases))
# network.input_layer.normalizer.adapt(train_inputs)  # normalizer is only adapted in the `NeuralNetwork.train()` method

train_predictions: NDArray = network.predict(train_inputs)
test_predictions: NDArray = network.predict(test_inputs)
print(f"Train Data Accuracy: {(100 * network.evaluate(train_targets, train_predictions)):.2f}%")
print(f"Test Data Accuracy: {(100 * network.evaluate(test_targets, test_predictions)):.2f}%")
