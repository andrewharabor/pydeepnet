
from time import time

from ._typing import *
from .error_metrics import ErrorMetric
from .layers import DenseLayer, DenseOutputLayer, InputLayer
from .optimizers import Optimizer

# Text colors for terminal output during training
BOLD_COLOR: str = "\033[1m"
BLUE_COLOR: str = "\033[94m"
CYAN_COLOR: str = "\033[96m"
GREEN_COLOR: str = "\033[92m"
PINK_COLOR: str = "\033[95m"
RED_COLOR: str = "\033[91m"
END_COLOR: str = "\033[0m"


class NeuralNetwork():
    input_layer: InputLayer
    hidden_layers: list[DenseLayer]
    output_layer: DenseOutputLayer
    optimizer: Optimizer
    error_metric: ErrorMetric | None

    _progress_bar_size: Int64 = Int64(25)

    def __init__(self, input_layer: InputLayer, hidden_layers: list[DenseLayer], output_layer: DenseOutputLayer, optimizer: Optimizer, error_metric: ErrorMetric | None) -> None:
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.optimizer = optimizer
        self.error_metric = error_metric
        for i, layer in enumerate(self.hidden_layers):
            if i == 0:
                layer.initialize_parameters(self.input_layer.nodes)
            else:
                layer.initialize_parameters(self.hidden_layers[i - 1].nodes)
        if len(self.hidden_layers) != 0:
            self.output_layer.initialize_parameters(self.hidden_layers[-1].nodes)
        else:
            self.output_layer.initialize_parameters(self.input_layer.nodes)
        self.optimizer.initialize(self.hidden_layers + [self.output_layer])

    def train(self, inputs: NDArray, targets: NDArray, epochs: Int64, mini_batch_size: Int64, verbose: bool = True) -> None:
        if inputs.shape[0] == 0 or targets.shape[0] == 0:
            raise ValueError("Inputs or targets arrays are empty")
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Inputs and targets arrays have different number of examples")
        if inputs.shape[1] != self.input_layer.nodes or targets.shape[1] != self.output_layer.nodes:
            raise ValueError("Inputs or targets arrays have incorrect number of features")
        if epochs <= 0:
            raise ValueError("Number of epochs is non-positive")
        if mini_batch_size <= 0:
            raise ValueError("Mini batch size is non-positive")
        start_time: Float64 = Float64(time())
        if self.input_layer.normalizer is not None:
            self.input_layer.normalizer.fit(inputs)
        for iteration in range(1, epochs + 1):
            random_indices: NDArray = np.random.permutation(inputs.shape[0])
            num_batches: Int64 = inputs.shape[0] // mini_batch_size + (1 if inputs.shape[0] % mini_batch_size != 0 else 0)
            for batch in range(1, num_batches + 1):
                # Prepare mini batch
                batch_indices: NDArray = random_indices[(batch - 1) * mini_batch_size:min(batch * mini_batch_size, inputs.shape[0])]
                batch_inputs: NDArray = inputs[batch_indices]
                batch_targets: NDArray = targets[batch_indices]
                examples: Int64 = Int64(batch_inputs.shape[0])
                # Initialization
                for layer in self.hidden_layers:
                    layer.initialize_gradients()
                self.output_layer.initialize_gradients()
                cost: Float64 = Float64(0.0)
                for i in range(examples):
                    # Forward propagation and compute cost
                    outputs: NDArray = self.input_layer.forward_propagation(batch_inputs[i])
                    for layer in self.hidden_layers:
                        outputs = layer.forward_propagation(outputs)
                        if layer.regularizer is not None:
                            cost += layer.regularizer.compute(layer.weights)
                    outputs = self.output_layer.forward_propagation(outputs)
                    cost += self.output_layer.cost_func.compute(outputs, batch_targets[i])
                    if self.output_layer.regularizer is not None:
                        cost += self.output_layer.regularizer.compute(self.output_layer.weights)
                    # Back propagation
                    derivative: NDArray = self.output_layer.start_back_propagation(batch_targets[i])
                    for layer in reversed(self.hidden_layers):
                        derivative = layer.back_propagation(derivative)
                # Update parameters
                for layer in self.hidden_layers:
                    layer.weights_gradient /= examples
                    layer.biases_gradient /= examples
                self.output_layer.weights_gradient /= examples
                self.output_layer.biases_gradient /= examples
                self.optimizer.update_parameters(self.hidden_layers + [self.output_layer])
                if verbose:
                    self._print_progress(self._progress_bar_size, Int64(iteration), epochs, Int64(batch), num_batches, cost)
            if verbose:
                print()
        if verbose:
            diff_secs: Float64 = Float64(time() - start_time)
            diff_mins: Int64 = Int64(diff_secs // 60)
            diff_hours: Int64 = diff_mins // 60
            diff_secs -= diff_mins * 60
            diff_mins %= 60
            print(f"Training completed in {BOLD_COLOR}{RED_COLOR}{diff_hours:.0f}h {diff_mins:.0f}m {diff_secs:.4f}s{END_COLOR}")

    def predict(self, inputs: NDArray) -> NDArray:
        if inputs.shape[0] == 0:
            raise ValueError("Inputs array is empty")
        if inputs.shape[1] != self.input_layer.nodes:
            raise ValueError("Inputs array has incorrect number of features")
        predictions: NDArray = np.zeros((inputs.shape[0], self.output_layer.nodes))
        for i in range(inputs.shape[0]):
            outputs: NDArray = self.input_layer.forward_propagation(inputs[i])
            for layer in self.hidden_layers:
                outputs = layer.forward_propagation(outputs)
            predictions[i] = self.output_layer.forward_propagation(outputs)
        return predictions

    def evaluate(self, predictions: NDArray, targets: NDArray) -> Float64:
        if predictions.shape[0] == 0 or targets.shape[0] == 0:
            raise ValueError("Predictions or targets arrays are empty")
        if predictions.shape[0] != targets.shape[0]:
            raise ValueError("Predictions and targets arrays have different number of examples")
        if self.error_metric is None:
            raise AttributeError("No error metric provided")
        return self.error_metric.compute(predictions, targets)

    def get_parameters(self) -> tuple[list[tuple[NDArray, NDArray]], tuple[NDArray, NDArray]]:
        hidden_layers_params: list[tuple[NDArray, NDArray]] = []
        for layer in self.hidden_layers:
            hidden_layers_params.append((layer.weights, layer.biases))
        return hidden_layers_params, (self.output_layer.weights, self.output_layer.biases)

    def set_parameters(self, hidden_layers_params: list[tuple[NDArray, NDArray]], output_layer_params: tuple[NDArray, NDArray]) -> None:
        if len(hidden_layers_params) != len(self.hidden_layers):
            raise ValueError("Number of hidden layers parameters is incorrect")
        for i, (weights, biases) in enumerate(hidden_layers_params):
            if weights.shape != self.hidden_layers[i].weights.shape or biases.shape != self.hidden_layers[i].biases.shape:
                raise ValueError("Shape of hidden layer parameters is incorrect")
            self.hidden_layers[i].weights = weights
            self.hidden_layers[i].biases = biases
        if output_layer_params[0].shape != self.output_layer.weights.shape or output_layer_params[1].shape != self.output_layer.biases.shape:
            raise ValueError("Shape of output layer parameters is incorrect")
        self.output_layer.weights = output_layer_params[0]
        self.output_layer.biases = output_layer_params[1]

    def _print_progress(self, size: Int64, iteration: Int64, epochs: Int64, batch: Int64, max_batch: Int64, cost: Float64) -> None:
        progress_bar: str = BOLD_COLOR + "[" + GREEN_COLOR + ("#" * round(batch / max_batch * size)) + END_COLOR + BOLD_COLOR + ("=" * round((max_batch - batch) / max_batch * size)) + "]" + END_COLOR
        print(f"Epoch {BLUE_COLOR}{iteration}{END_COLOR}/{epochs} ({(iteration / epochs * 100):.2f}%)     {progress_bar} Batch {CYAN_COLOR}{batch}{END_COLOR}/{max_batch} ({(batch / max_batch * 100):.2f}%)     Cost = {PINK_COLOR}{cost}{END_COLOR}", end="\r")
