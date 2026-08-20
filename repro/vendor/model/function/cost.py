from abc import ABC, abstractmethod
import numpy
import torch
import torch.nn.functional as F

from model.function.interaction import Function, QFunction
from model.variable.parameter import Bias, DenseWeight
from model.variable.layer import Layer



class CostFunction(ABC):
    """Abstract class for cost functions

    Attributes
    ----------
    _num_classes (int): the number of categories in the classification task
    _label (Tensor): label tensor. Shape is (batch size,). Type is int.
    _target (Tensor): one-hot-label tensor. Shape is (batch size, output_shape). Type is float.

    Methods
    -------
    set_target(label)
        Set target values
    error_fn()
        Computes the error value for the current state configuration
    top_five_error_fn()
        Computes the top-5 error rate for the current output configuration.
    _get_output()
        Returns the output layer, or the prediction
    """

    def __init__(self, num_classes):
        """Initializes an instance of CostFunction

        Args:
            num_classes (int): number of categories in the classification task
        """

        # FIXME: should be using the init method of Function

        self._num_classes = num_classes  # number of categories in the classification task

        self._label = None  # FIXME
        self._target = None  # FIXME

    def set_target(self, label):
        """Set label and target values

        Args:
            label: Tensor of shape (batch size,) and type int. Labels associated to the inputs in the batch.
        """

        # TODO: check that the batch_size of the labels is the same as that of the output layer

        output = self._get_output()  # the output layer's state
        device = output.device  # device on which the output layer Tensor is, and on which we put the layer and target Tensors
        self._label = label.to(device)
        self._target = F.one_hot(self._label, num_classes=self._num_classes).type(torch.float32)  # convert the label into its one-hot code

    def error_fn(self):
        """Returns the error value for the current output configuration.

        Returns:
            Tensor of shape (batch_size,) and type bool. Vector of error values for each of the examples in the current mini-batch
        """

        output = self._get_output()  # state of output layer
        prediction = torch.argmax(output, dim=1)  # the predicted category is the index of the output unit that has the highest value
        label = self._label
        return torch.ne(prediction, label)

    def top_five_error_fn(self):
        """Returns the top-5 accuracy for the current output configuration.

        Returns:
            Tensor of shape (batch_size,) and type bool. Vector of top-5 error values for each of the examples in the current mini-batch
        """

        output = self._get_output()  # state of output layer
        _, indices = torch.topk(output, 5, dim=1)  # top-5 categories indices of the output unit
        label = self._label.unsqueeze(-1)
        return torch.all(torch.ne(indices, label), dim=1)

    @abstractmethod
    def _get_output(self):
        """Returns the output layer, or the prediction"""
        pass



class SquaredError(CostFunction, QFunction):
    """Class for the squared error cost function between the output layer and the target layer.

    Methods
    -------
    eval()
        Returns the squared error between the output layer and the target
    """

    def __init__(self, layer):
        """Initializes an instance of SquaredError

        Args:
            layer (Layer): the layer playing the role of `output layer', or prediction
        """

        self._layer = layer

        num_classes = layer.state.shape[1]  # number of categories in the classification task
        CostFunction.__init__(self, num_classes)

        Function.__init__(self, [layer], [])

    def eval(self):
        """Returns the cost value (the squared error) of the current state configuration.

        Returns:
            Tensor of shape (batch_size,) and type float32. Vector of cost values for each of the examples in the current mini-batch
        """

        output = self._layer.state  # state of output layer: shape is (batch_size, num_classes)
        target = self._target  # desired output: shape is (batch_size, num_classes)
        return 0.5 * ((output - target) ** 2).sum(dim=1)  # Vector of shape (batch_size,)

    def _get_output(self):
        """Returns the output layer, or the prediction"""
        return self._layer.state

    def grad_layer_fn(self, layer):
        """Overrides the default implementation of Function"""
        dictionary = {self._layer: self._grad_layer}
        return dictionary[layer]

    def a_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer: self._a_coef,
            }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer: self._b_coef,
            }
        return dictionary[layer]

    def _grad_layer(self):
        """Returns the gradient of the cost function wrt the outputs"""
        return self._layer.state - self._target

    def _a_coef(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution on layer_pre
        """

        return 0.5

    def _b_coef(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution on layer_pre
        """

        return - self._target

    def __str__(self):
        return 'Squared Error (MSE)'


    # =====================
# ---------------------
# Paired-outputs squared error (Kendall-style)
# ---------------------


class SquaredErrorPairedOutputs(CostFunction, QFunction):
    """
    MSE on pairwise differences:
      score_k = y_{2k} - y_{2k+1}
      loss = 0.5 * || score - target ||^2
    Targets can be one-hot (B, K) or float scores.
    """

    def __init__(self, layer: 'Layer', num_classes: int):
        self._layer = layer
        self._num_classes = int(num_classes)
        self._label = None        # (B,)
        self._target = None       # (B, K)
        # Initialize as a (cost) Function of just the output layer
        Function.__init__(self, [layer], [])

    # --- required by CostFunction base ---
    def _get_output(self):
        # Return the raw 2K outputs; CostFunction.error_fn/top_five use this.
        return self._layer.state

    # --- targets ---
    def set_target(self, label):
        """
        Provide either label (B,) with class indices or target (B, K) scores.
        """
        output = self._get_output()  # the output layer's state
        device = output.device  # device on which the output layer Tensor is, and on which we put the layer and target Tensors
        self._label = label.to(device)
        self._target = F.one_hot(self._label, num_classes=self._num_classes).type(torch.float32)  # convert the label into its one-hot code

    def _scores(self, y: torch.Tensor) -> torch.Tensor:
        B, N = y.shape
        K = self._num_classes
        assert N == 2 * K, f"SquaredErrorPairedOutputs: expected {2*K} outputs, got {N}"
        y2 = y.view(B, K, 2)
        return y2[..., 0] - y2[..., 1]  # (B, K)

    def _target_vec(self, device, dtype):
        if self._label is None:
            raise RuntimeError(
                "SquaredErrorPairedOutputs: targets not set. "
                "Call set_target(label=...)."
            )

        labels = self._label.to(device=device, dtype=torch.long).view(-1)
        return F.one_hot(labels, num_classes=self._num_classes).to(dtype=dtype)


    # --- loss value ---
    def eval(self):
        y = self._layer.state      # (B, 2K)
        s = self._scores(y)        # (B, K)
        t = self._target_vec(y.device, y.dtype)
        diff = s - t
        return 0.5 * (diff * diff).sum(dim=1)

    # --- prediction / error (paired argmax) ---
    def error_fn(self):
        if self._label is None:
            raise RuntimeError("SquaredErrorPairedOutputs: label not set. Call set_target(label=...).")
        y = self._layer.state
        scores = self._scores(y)
        pred = torch.argmax(scores, dim=1)
        # compare on the same device
        labels = self._label.to(device=y.device)
        return torch.ne(pred, labels)

    def top_five_error_fn(self):
        if self._label is None:
            raise RuntimeError("SquaredErrorPairedOutputs: label not set. Call set_target(label=...).")
        y = self._layer.state
        scores = self._scores(y)
        k = min(5, self._num_classes)
        topk = torch.topk(scores, k=k, dim=1).indices
        labels = self._label.to(device=y.device)
        return ~ (topk == labels.view(-1, 1)).any(dim=1)



    def a_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer: self._a_coef,
            }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer: self._b_coef,
            }
        return dictionary[layer]



    def _a_coef(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution on layer_pre
        """

        return 0.5

    def _b_coef(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution on layer_pre
        """

        y = self._layer.state              # (B, 2K)
        B, N = y.shape
        K = self._num_classes
        assert N == 2 * K, f"Expected 2K outputs, got {N}"
        y2 = y.view(B, K, 2)               # (B, K, 2)
        y_even = y2[..., 0]                # (B, K) -> index 2k
        y_odd  = y2[..., 1]                # (B, K) -> index 2k+1
        t = self._target_vec(y.device, y.dtype)  # (B, K)
        #s_k = y_{2k} – y_{2k+1}
        #L = 0.5 * sum_k (s_k – t_k)^2

        b_even = -(y_odd + t)              # shape (B, K)
        b_odd  =  (t - y_even)             # shape (B, K)

        b = torch.empty_like(y)
        b[:, 0::2] = b_even
        b[:, 1::2] = b_odd
        return b


    def __str__(self):
        return 'Squared Error (MSE)'


    # --- gradient for EP/CpL (dL/dvar) ---
    def _grad(self, var, mean: bool):
        """
        Gradient wrt 'var'.
        If var is the output layer, returns dL/dy (shape (B, 2K)).
        """
        if isinstance(var, Layer) and var is self._layer:
            y = self._layer.state
            s = self._scores(y)                # (B, K)
            t = self._target_vec(y.device, y.dtype)
            g = (s - t)                        # dL/ds (B,K)

            # Map to 2K outputs:
            # s_k = y_{2k} - y_{2k+1} => dL/d y_{2k} = g_k, dL/d y_{2k+1} = -g_k
            B, K = g.shape
            dy = torch.zeros(B, 2*K, device=y.device, dtype=y.dtype)
            dy[:, 0::2] = g
            dy[:, 1::2] = -g

            return dy.mean(dim=0) if mean else dy

        # Default: no explicit param dependence
        if hasattr(var, "state"):
            z = var.state
            return z.new_zeros(z.shape).mean(dim=0) if mean else z.new_zeros(z.shape)

        return torch.tensor(0., device=self._layer.state.device, dtype=self._layer.state.dtype)
