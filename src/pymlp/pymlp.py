
from .error_metrics import ErrorMetric
from .layers import DenseLayer, DenseOutputLayer, InputLayer
from .optimizers import Optimizer
from .typing import *


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
        if self.input_layer.normalizer is not None:
            self.input_layer.normalizer.adapt(inputs)
        for iteration in range(1, epochs + 1):
            rand_indices: NDArray = np.random.permutation(inputs.shape[0])
            max_batch: Int64 = inputs.shape[0] // mini_batch_size + (1 if inputs.shape[0] % mini_batch_size != 0 else 0)
            for batch in range(1, max_batch + 1):
                # Prepare mini batch
                batch_indices: NDArray = rand_indices[(batch - 1) * mini_batch_size:min(batch * mini_batch_size, inputs.shape[0])]
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
                    self._print_progress(self._progress_bar_size, Int64(iteration), epochs, Int64(batch), max_batch, cost)
            if verbose:
                print()

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

    def _print_progress(self, size: Int64, iteration: Int64, epochs: Int64, batch: Int64, max_batch: Int64, cost: Float64) -> None:
        progress_bar: str = "[" + ("#" * round(batch / max_batch * size)) + ("=" * round((max_batch - batch) / max_batch * size)) + "]"
        print(f"Epoch {iteration}/{epochs} ({(iteration / epochs * 100):.2f}%)     {progress_bar} Batch {batch}/{max_batch} ({(batch / max_batch * 100):.2f}%)     Cost = {cost}", end="\r")
