
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

    def train(self, inputs: NDArray, targets: NDArray, epochs: int) -> None:
        examples: int = inputs.shape[0]
        if self.input_layer.normalizer is not None:
            self.input_layer.normalizer.adapt(inputs)
        for _ in range(1, epochs + 1):
            for layer in self.hidden_layers:
                layer.initialize_gradients()
            self.output_layer.initialize_gradients()
            cost: Float64 = Float64(0.0)
            for i in range(examples):
                # Forward propagation and compute cost
                outputs: NDArray = self.input_layer.forward_propagation(inputs[i])
                for layer in self.hidden_layers:
                    outputs = layer.forward_propagation(outputs)
                    if layer.regularizer is not None:
                        cost += layer.regularizer.compute(layer.weights)
                outputs = self.output_layer.forward_propagation(outputs)
                cost += self.output_layer.cost_func.compute(outputs, targets[i])
                if self.output_layer.regularizer is not None:
                    self.output_layer.regularizer.compute(self.output_layer.weights)
                # Back propagation
                derivative: NDArray = self.output_layer.start_back_propagation(targets[i])
                for layer in reversed(self.hidden_layers):
                    derivative = layer.back_propagation(derivative)
            # Update parameters
            for layer in self.hidden_layers:
                layer.weights_gradient /= examples
                layer.biases_gradient /= examples
            self.output_layer.weights_gradient /= examples
            self.output_layer.biases_gradient /= examples
            self.optimizer.update_parameters(self.hidden_layers + [self.output_layer])

    def predict(self, inputs: NDArray) -> NDArray:
        predictions: NDArray = np.zeros((inputs.shape[0], self.output_layer.nodes))
        for i in range(inputs.shape[0]):
            outputs: NDArray = self.input_layer.forward_propagation(inputs[i])
            for layer in self.hidden_layers:
                outputs = layer.forward_propagation(outputs)
            outputs = self.output_layer.forward_propagation(outputs)
            predictions[i] = outputs
        return predictions

    def evaluate(self, predictions: NDArray, targets: NDArray) -> Float64:
        if self.error_metric is None:
            raise AttributeError("No error metric provided")
        return self.error_metric.compute(predictions, targets)
