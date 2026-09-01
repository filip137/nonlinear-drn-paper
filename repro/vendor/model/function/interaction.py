from abc import ABC, abstractmethod
import copy
import torch
import torch.nn.functional as F

from model.amplification import amp_ratio, tensor_like
from model.variable.layer import layer_index


VALID_BIAS_SCALE_MODES = ("legacy", "constant")
VALID_BIAS_INTERACTION_TYPES = ("linear", "resistive", "quadratic")


def resolve_bias_interaction_scale(layer, voltage_amp, current_amp, mode="legacy"):
    """Return the scale applied to a layer's linear bias interaction."""

    if mode == "legacy":
        return amp_ratio(current_amp, voltage_amp, layer_index(layer) - 1)
    if mode == "constant":
        return 1.0
    raise ValueError(
        f"Unknown bias_scale_mode '{mode}'. Expected one of {VALID_BIAS_SCALE_MODES}."
    )


def build_bias_interaction(
    layer,
    bias,
    voltage_amp,
    current_amp,
    *,
    scale_mode="legacy",
    interaction_type="linear",
):
    """Create the configured bias interaction with the standard amplification scaling."""

    scale = resolve_bias_interaction_scale(
        layer,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        mode=scale_mode,
    )
    if interaction_type == "linear":
        return BiasInteraction(
            layer,
            bias,
            scale=scale,
            voltage_amp=voltage_amp,
            current_amp=current_amp,
            scale_mode=scale_mode,
        )
    if interaction_type in ("resistive", "quadratic"):
        return ResistiveBiasInteraction(
            layer,
            bias,
            scale=scale,
            voltage_amp=voltage_amp,
            current_amp=current_amp,
            scale_mode=scale_mode,
        )
    raise ValueError(
        "Unknown bias_interaction_type "
        f"'{interaction_type}'. Expected one of {VALID_BIAS_INTERACTION_TYPES}."
    )



class Function(ABC):
    """Abstract class for functions

    A function object is defined by its argument variables (layers z_j and parameters theta_k) and by its function E_i({z_j},{theta_k})

    Attributes
    ----------
    device (str): The device where the tensors of the function are set. Either 'cpu' or 'cuda'

    Methods
    -------
    params()
        Returns the list of parameter variables of the function
    layers()
        Returns the list of layer variables of the function
    eval():
        Returns the value of the function (E_i) evaluated at the current state of the variables
    grad_layer_fn(layer):
        Returns the gradient function wrt the layer ({} -> dE_i/dz_j)
    grad_param_fn(param):
        Returns the gradient function wrt the parameter ({} -> dE_i/dtheta_k)
    second_fn(param):
        Returns the `second function' wrt the parameter (direction -> d^2E_i / dtheta_k ds * direction)
    set_device(device)
        Set the tensors of the function on a given device
    to(device):
        Returns a copy of the Function object, and sets the tensors on the desired device
    save(path)
        Saves the parameters of the function
    load(path)
        Loads the parameters of the function
    """

    def __init__(self, layers, params):
        """Constructor of Function

        Args:
            layers (list of Layer): the layers involved in the function
            params (list of Parameter): the parameters involved in the function
        """

        self._layers = layers
        self._params = params

        self._device = None  # FIXME: this attribute should be removed and part of the Variable class only

        # self._grad_fns = dict()
        # for layer in layers: self._grad_fns[layer] = lambda: self._grad(layer, mean=False)
        # for param in params: self._grad_fns[param] = lambda: self._grad(param, mean=True)

    def params(self):
        """Returns the list of parameter variables of the function"""
        return self._params

    def layers(self):
        """Returns the list of layer variables of the function"""
        return self._layers

    @abstractmethod
    def eval(self):
        """Returns the value of the function evaluated at the current state of the variables

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the value of the function wrt an example in the current mini-batch
        """
        pass

    def grad_layer_fn(self, layer):
        """Returns the gradient function wrt the given layer

        Default implementation, valid for any layer and any function

        Args:
            layer (Layer): the layer whose gradient wrt the function we want to compute

        Returns:
            function that computes the gradient of the corresponding layer
        """
        return lambda: self._grad(layer, mean=False)
        # return self._grad_fns[layer]

    def grad_param_fn(self, param):
        """Returns the gradient function wrt the given parameter

        Default implementation, valid for any parameter and any function

        Args:
            param (Parameter): the parameter whose gradient wrt the function we want to compute

        Returns:
            function that computes the gradient of the corresponding layer
        """
        return lambda: self._grad(param, mean=True)
        # return self._grad_fns[param]

    def second_fn(self, param):
        """Returns the function: direction -> d^2E / dtheta ds * direction, wrt the parameter (theta)

        Default implementation, valid for any parameter, any layers, and any function

        Returns:
            functions that takes in a dictionary of Tensors of shape (batch_size, layer_shape), and returns a Tensor of shape param_shape
        """
        return lambda direction: self._second_derivative(param, direction)

    def _grad(self, variable, mean=False):
        """Returns the variable's gradient wrt the function

        Implementation valid for any variable (layer or parameter) and any function
        Implementation valid whether the variable's requires_grad attribute is set to True or False

        Args:
            variable (Variable): the variable whose gradient we want to compute
            mean (bool, optional): whether we compute the gradient of the mean of the function, or the sum of the function (over the mini-batch of examples). Default: False.

        Returns:
            Tensor: the gradient of the function wrt the variable. Shape is (batch_size, layer_shape) if the variable is a Layer, or param_shape if it is a Parameter. Type is float32
        """

        if variable.state.requires_grad == False:
            variable.state.requires_grad = True
            value = torch.mean( self.eval() ) if mean else torch.sum( self.eval() )
            grad = torch.autograd.grad(value, variable.state)[0]
            variable.state.requires_grad = False

            # FIXME: here for this piece of code to run correctly with DNNs trained by Backprop, we need
            # grad = torch.autograd.grad(value, variable.state, create_graph=True)[0]
            # How to solve this?

        else:
            # if the variable already has its requires_grad attribute set to True, we create the graph of derivatives as we compute the gradient of the energy function
            value = torch.mean( self.eval() ) if mean else torch.sum( self.eval() )
            grad = torch.autograd.grad(value, variable.state, create_graph=True)[0]

        return grad

    '''def grad(self, variables):
        """Returns the gradient of the function wrt the variables

        Implementation valid for any variables (layers or parameters) and any function
        Implementation valid whether the variables' requires_grad attributes are set to True or False

        Args:
            variables (list of Tensor): the variables whose gradient we want to compute

        Returns:
            Tensor: the gradient of the function wrt the variable. Shape is (batch_size, layer_shape) if the variable is a Layer, or param_shape if it is a Parameter. Type is float32
        """

        bools = [variable.requires_grad for variable in variables]

        for variable in variables: variable.requires_grad = True

        if not any(bools):
            value = torch.mean( self.eval() )
            grad = torch.autograd.grad(value, variables)
        else:
            # if the variable already has its requires_grad attribute set to True, we create the graph of derivatives as we compute the gradient of the energy function
            value = torch.mean( self.eval() )
            grad = torch.autograd.grad(value, variables, create_graph=True)

        for variable, boolean in zip(variables, bools): variable.requires_grad = boolean

        return grad'''

    def _second_derivative(self, param, direction):
        """Returns the param's gradient wrt the directional derivative of the function (in the direction `direction')

        Implementation valid for any parameter and any function
        Implementation valid whether the parameter's requires_grad attribute is set to True or False

        This method is only used in the context of EquilibriumProp. Therefore the requires_grad attributes of the layers and params are set to False when calling this method.
        The requires_grad attributes of direction must also be set to False.

        Args:
            param (Parameter): the parameter whose gradient we want to compute
            direction (dict of Tensor): the direction in the state space in which we take the directional derivative of the function

        Returns:
            Tensor: the gradient wrt the parameter of the directional derivative of the function (in the direction state). Shape is param_shape. Type is float32
        """

        # TODO: I think we only need to set param.state.requires_grad = True instead of setting all params' requires_grad attribute to True

        for layer in self._layers: layer.state.requires_grad = True
        for param in self._params: param.state.requires_grad = True
        value = torch.mean( self.eval() )
        layer_grads = torch.autograd.grad(value, [layer.state for layer in self._layers], create_graph=True)
        direction = [direction[layer.name] for layer in self._layers]
        param_grad = torch.autograd.grad(layer_grads, param.state, grad_outputs = direction)[0]
        for layer in self._layers: layer.state.requires_grad = False
        for param in self._params: param.state.requires_grad = False

        return param_grad

    def set_device(self, device):
        """Set the tensors of the function on a given device

        Args:
            device (str): the name of the device (e.g. 'cpu' or 'cuda')
        """

        self._device = device

        for layer in self._layers: layer.set_device(device)
        for param in self._params: param.set_device(device)

    def to(self, device):
        """Returns a copy of the Function object, and sets the tensors on the desired device

        Args:
            device (str): the name of the device (e.g. 'cpu' or 'cuda')
        """
        function = copy.deepcopy(self)
        function.set_device(device)
        return function

    def save(self, path):
        """Saves the function parameters

        Args:
            path (str): path where to save the function's parameters
        """

        # FIXME: the parameters of the Readout cost function won't be saved

        params = [param.state for param in self._params]
        torch.save(params, path)

    def load(self, path):
        """Loads the function parameters

        Args:
            path (str): path where to load the function's parameters from
        """

        # FIXME: the parameters of the Readout cost function won't be loaded

        states = torch.load(
            path,
            map_location=torch.device(self._device),
            weights_only=True,
        )
        if not isinstance(states, (list, tuple)):
            raise ValueError(
                "Expected checkpoint payload to be a list or tuple of parameter tensors. "
                f"Provided value: {type(states).__name__}."
            )
        if len(states) != len(self._params):
            raise ValueError(
                "Expected checkpoint parameter count to match the constructed model "
                f"exactly: expected {len(self._params)}, got {len(states)}."
            )

        # Validate the complete checkpoint before assigning any tensor. A
        # partial or malformed checkpoint must never leave random parameters in
        # an otherwise apparently loaded model.
        for index, (param, state) in enumerate(zip(self._params, states)):
            if not isinstance(state, torch.Tensor):
                raise ValueError(
                    f"Expected checkpoint parameter {index} to be a tensor. "
                    f"Provided value: {type(state).__name__}."
                )
            if tuple(state.shape) != tuple(param.state.shape):
                raise ValueError(
                    f"Expected checkpoint parameter {index} shape "
                    f"{tuple(param.state.shape)}, got {tuple(state.shape)}."
                )
            if state.dtype != param.state.dtype:
                raise ValueError(
                    f"Expected checkpoint parameter {index} dtype "
                    f"{param.state.dtype}, got {state.dtype}."
                )
            if not torch.isfinite(state).all():
                raise ValueError(
                    f"Expected checkpoint parameter {index} to contain only finite values."
                )
            minimum = param.min_cond
            maximum = param.max_cond
            if minimum is None and getattr(param, "_non_negative", False):
                minimum = 0.0
            if minimum is not None and torch.any(state < minimum):
                raise ValueError(
                    f"Expected checkpoint parameter {index} to satisfy its configured "
                    f"minimum {minimum}."
                )
            if maximum is not None and torch.any(state > maximum):
                raise ValueError(
                    f"Expected checkpoint parameter {index} to satisfy its configured "
                    f"maximum {maximum}."
                )
        for param, state in zip(self._params, states):
            param.state = state.detach()



class QFunction(Function, ABC):
    """Abstract class for Q-functions

    A Q-function is a function E such that for each layer z, E(z) is a quadratic function of z, i.e. of the form
    E(z) = a z^2 + b z + c
    for some coefficients a, b and c. We say that a is the 'quadratic coefficient' of z, and b is the 'linear coefficient' of z.

    Methods
    -------
    a_coef_fn(layer):
        Returns a function that computes the coefficient a of the layer
    b_coef_fn(layer):
        Returns a function that computes the coefficient b of the layer
    """

    @abstractmethod
    def a_coef_fn(self, layer):
        """Returns a function that computes the coefficient a of the layer

        The function E is quadratic in each layer z, i.e. of the form E(z) = a z^2 + b z + c

        Args:
            layer (Layer): the layer whose coefficient a we want to compute

        Returns:
            function that computes the coefficient a of the layer
        """
        pass

    @abstractmethod
    def b_coef_fn(self, layer):
        """Returns a function that computes the coefficient b of the layer

        The function E is quadratic in each layer z, i.e. of the form E(z) = a z^2 + b z + c

        Args:
            layer (Layer): the layer whose coefficient b we want to compute

        Returns:
            function that computes the coefficient b of the layer
        """
        pass

    # TODO: override grad_layer_fn
    '''def grad_layer_fn(self, layer):
        """Returns the gradient of the function wrt the layer, i.e. dE/dz, where z is the layer

        Overrides the default implementation of the class Function

        By assumption, the function E as a function of z is of the form E(z) = a * z^2 + b * z + c.
        So, the gradient can be calculated as dE/dz = 2 a * z + b

        Args:
            layer (Layer): the layer whose gradient we want to compute

        Returns:
            Tensor of shape (batch_size, layer_shape). Type is float32
        """
        a_fn = self.a_coef_fn(layer)
        b_fn = self.b_coef_fn(layer)
        return lambda: 2. * a_fn() * layer.state + b_fn()  # tensor of size (batch_size, layer_shape)'''



class LFunction(QFunction, ABC):
    """Abstract class for L-functions

    A L-function with respect to a set of variables V is a function which is linear in z for each variable z in V.
    That is, for each variable z in V, the function E as a function of z is of the form
    E(z) = b z + c
    for some coefficients b and c.
    Therefore, a L-function wrt V is also a Q-function wrt to V.
    """

    def a_coef_fn(self, layer):
        """Returns the function that computes the coefficient a for a given layer
        The coefficient a of a multi-linear function is 0, so we return None."""
        return None

    def b_coef_fn(self, layer):
        """Returns the function that computes the coefficient b for a given layer

        The function is linear in each layer z, i.e. of the form E(z) = b z + c
        The linear coefficient of z is b, which is dE/dz"""
        return self.grad_layer_fn(layer)



class BiasInteraction(LFunction):
    """Interaction of the Bias

    A bias interaction is defined between a layer and its corresponding bias variable

    Attributes
    ----------
    _layer (Layer): the layer involved in the interaction
    _bias (Bias): the layer's bias
    """

    def __init__(self, layer, bias, scale=1.0, voltage_amp=None, current_amp=None, scale_mode=None):
        """Initializes an instance of BiasInteraction

        Args:
            layer (Layer): the layer involved in the interaction
            bias (Bias): the layer's bias
            scale (float): optional scaling factor applied to the bias contribution
        """

        self._layer = layer
        self._bias = bias
        self._scale = scale
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._scale_mode = scale_mode

        LFunction.__init__(self, [layer], [bias])

    def _scale_value(self):
        if self._scale_mode is None:
            return self._scale
        return resolve_bias_interaction_scale(
            self._layer,
            voltage_amp=self._voltage_amp,
            current_amp=self._current_amp,
            mode=self._scale_mode,
        )

    def eval(self):
        """Energy term of the bias.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        return - self._scale_value() * self._layer.state.mul(self._bias.get()).flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        """Returns the function that computes the gradient wrt a given layer"""
        dictionary = {self._layer: self._b_coef_layer}
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._bias: self._grad_bias}
        return dictionary[param]

    def _b_coef_layer(self):
        """Returns the linear influence of the bias on the layer's state.

        Returns:
            Tensor of shape (batch_size, layer_shape) and type float32: the linear contribution
        """

        return - self._scale_value() * self._bias.get()

    def _grad_bias(self):
        """Returns the interaction's gradient wrt the bias"""

        # FIXME: there is a problem here, e.g. if the layer has shape (2, 1024) and the associated bias also has shape (2, 1024)
        #commented out so that it works with the CNN


        coef = - self._scale_value() * self._layer.state.mean(dim=0)

        #if len(coef.shape) > 1:
            #dims = tuple(range(1, len(coef.shape)))
            #coef = coef.sum(dim=dims, keepdim=True)

        return coef



class ResistiveBiasInteraction(QFunction):
    """Quadratic bias interaction that contributes to a (z^2 term) only."""

    def __init__(self, layer, bias, scale=1.0, voltage_amp=None, current_amp=None, scale_mode=None):
        """Initializes an instance of ResistiveBiasInteraction.

        Args:
            layer (Layer): the layer involved in the interaction
            bias (Bias): the layer's bias (used as a quadratic coefficient)
            scale (float): optional scaling factor applied to the bias term
        """

        self._layer = layer
        self._bias = bias
        self._scale = scale
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._scale_mode = scale_mode

        QFunction.__init__(self, [layer], [bias])

    def _scale_value(self):
        if self._scale_mode is None:
            return self._scale
        return resolve_bias_interaction_scale(
            self._layer,
            voltage_amp=self._voltage_amp,
            current_amp=self._current_amp,
            mode=self._scale_mode,
        )

    def eval(self):
        """Energy term of the quadratic bias.

        Returns:
            Vector of size (batch_size,) and of type float32.
        """

        coef = self._bias.get() * self._scale_value()
        return self._layer.state.pow(2).mul(coef).flatten(start_dim=1).sum(dim=1)

    def a_coef_fn(self, layer):
        """Overrides the default implementation of QFunction."""
        dictionary = {self._layer: self._a_coef_layer}
        return dictionary[layer]

    def b_coef_fn(self, layer):
        """Overrides the default implementation of QFunction."""
        dictionary = {self._layer: self._b_coef_layer}
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function."""
        dictionary = {self._bias: self._grad_bias}
        return dictionary[param]

    def _a_coef_layer(self):
        """Returns the quadratic influence of the bias on the layer's state."""
        return (self._bias.get() * self._scale_value()).unsqueeze(0)

    def _b_coef_layer(self):
        """Returns zero linear influence on the layer's state."""
        return torch.zeros_like(self._layer.state)

    def _grad_bias(self):
        """Returns the interaction's gradient wrt the bias."""
        return (self._layer.state.pow(2) * self._scale_value()).mean(dim=0)


class HardSigmoidNonLinearInteraction(Function):
    """Piecewise-quadratic clamp with separate on/off conductances."""

    def __init__(self, layer, params, voltage_amp, current_amp):
        self._layer = layer
        index = layer_index(layer)
        self._scale_exponent = index - 1
        self._g_on_base = params.get("g_on")
        self._g_off_base = params.get("g_off")
        self.v_min = params.get("v_min")
        self.v_max = params.get("v_max")
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._g_on = self._g_on_base * self._scale_value()
        self._g_off = self._g_off_base * self._scale_value()

        Function.__init__(self, [layer], [])

    def _scale_value(self):
        return amp_ratio(self._current_amp, self._voltage_amp, self._scale_exponent)

    def _conductances(self):
        scale = self._scale_value()
        return self._g_on_base * scale, self._g_off_base * scale

    def eval(self):
        """Energy term of the non-linearity."""
        v = self._layer.state
        g_on, g_off = self._conductances()
        off_mask = (v >= self.v_min) & (v <= self.v_max)
        pos_on_mask = v > self.v_max
        neg_on_mask = v < self.v_min

        energy_off = 0.5 * g_off * (v ** 2) * off_mask
        energy_on = torch.zeros_like(v)
        # Integrate the piecewise current continuously from 0:
        # for v > v_max, carry the off-region contribution up to v_max;
        # for v < v_min, carry the off-region contribution down to v_min.
        energy_on[pos_on_mask] = (
            0.5 * g_off * (self.v_max ** 2)
            + 0.5 * g_on * (v[pos_on_mask] - self.v_max) ** 2
        )
        energy_on[neg_on_mask] = (
            0.5 * g_off * (self.v_min ** 2)
            + 0.5 * g_on * (v[neg_on_mask] - self.v_min) ** 2
        )

        energy = energy_off + energy_on
        return energy.flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        """Returns the gradient function wrt the given layer."""
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._grad_layer

    def _grad_layer(self):
        """Returns dE/dv for the associated non-linearity."""
        v = self._layer.state
        g_on, g_off = self._conductances()
        grad = torch.zeros_like(v)

        off_mask = (v >= self.v_min) & (v <= self.v_max)
        pos_on_mask = v > self.v_max
        neg_on_mask = v < self.v_min

        grad[off_mask] = g_off * v[off_mask]
        grad[pos_on_mask] = g_on * (v[pos_on_mask] - self.v_max)
        grad[neg_on_mask] = g_on * (v[neg_on_mask] - self.v_min)

        return grad

    def a_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)

    def b_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)


class LpwNonLinearInteraction(Function):
    """LPW diode interaction: quadratic penalty for sign-violations beyond v_off."""

    def __init__(self, layer, params, voltage_amp, current_amp):
        self._layer = layer
        index = layer_index(layer)
        self._scale_exponent = index - 1
        self._g_base = params.get("diode_conductance")
        self._v_off = params.get("v_off", 0.0)
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._g = self._g_base * self._scale_value()

        Function.__init__(self, [layer], [])

    def _scale_value(self):
        return amp_ratio(self._current_amp, self._voltage_amp, self._scale_exponent)

    def _conductance(self):
        return self._g_base * self._scale_value()

    def _split(self, v):
        dim = v.shape[1] // 2
        return v[:, :dim], v[:, dim:]

    def eval(self):
        v = self._layer.state
        exc, inh = self._split(v)
        g = self._conductance()
        exc_excess = torch.clamp(exc - self._v_off, min=0.0)
        inh_excess = torch.clamp(-inh - self._v_off, min=0.0)
        energy = 0.5 * g * (exc_excess ** 2)
        energy = energy + 0.5 * g * (inh_excess ** 2)
        return energy.flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._grad_layer

    def _grad_layer(self):
        v = self._layer.state
        exc, inh = self._split(v)
        g = self._conductance()
        exc_excess = torch.clamp(exc - self._v_off, min=0.0)
        inh_excess = torch.clamp(-inh - self._v_off, min=0.0)
        grad_exc = g * exc_excess
        grad_inh = -g * inh_excess
        return torch.cat((grad_exc, grad_inh), dim=1)

    def a_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)

    def b_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)


class DoubleQuadraticNonLinearInteraction(Function):
    """Interaction with the quadrati dissipative non-linearity

    A nonlinear interaction is defined between the voltage at the hidden layer and ground connected by the nonlineairty

    Attributes
    ----------
    layer (Layer): the layer involved in the interaction
    non_lineartiy (string) : type of the non linearity involved in the interaction
    c (float) : diode's conductances
    v_off, v_on (float) : the voltage at which the reverse and the forward diode turn on
    """
    def __init__(self, layer, params, voltage_amp, current_amp):
        self._layer = layer
        index = layer_index(layer)
        self._scale_exponent = index - 1
        self._c_base = params.get("diode_conductance", params.get("c"))
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self.c = self._c_base * self._scale_value()
        self.v_off = params.get("v_off", 0.0)

        Function.__init__(self, [layer], [])

    def _scale_value(self):
        return amp_ratio(self._current_amp, self._voltage_amp, self._scale_exponent)

    def _conductance(self):
        return self._c_base * self._scale_value()


    def eval(self):
        """Energy term of the non-linearity.
        if abs(v) > v_off = v_on
            E = c/3 * abs(v)**3 ()
        else
            0
        Returns:
            Tensor of shape (batch_size, layer_shape) and type float32: the cubic contribution
        """
        v = self._layer.state
        c = self._conductance()
        excess = torch.clamp(torch.abs(v) - self.v_off, min=0.0)
        energy = (c / 3.0) * excess ** 3
        return energy.flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        """Returns the gradient function wrt the given layer."""
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._grad_layer

    def _grad_layer(self):
        """Returns dE/dv for the associated non-linearity."""
        v = self._layer.state
        c = self._conductance()
        excess = torch.clamp(torch.abs(v) - self.v_off, min=0.0)
        grad = c * excess ** 2 * torch.sign(v)
        return grad

    def a_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)

    def b_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)


class DoubleExponentialNonLinearInteraction(Function):
    """Interaction with the quadrati dissipative non-linearity

    A nonlinear interaction is defined between the voltage at the hidden layer and ground connected by the nonlineairty
    We scale the parameter I_s in the case there is amplification
    Attributes
    ----------
    layer (Layer): the layer involved in the interaction
    non_lineartiy (string) : type of the non linearity involved in the interaction
    """
    def __init__(self, layer, params, voltage_amp, current_amp):
        self._layer = layer
        index = layer_index(layer)
        self._scale_exponent = index - 1
        self._I_s_base = params.get("I_s")
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self.I_s = self._I_s_base * self._scale_value()
        self.vt = params.get("V_t")
        self.v_off = params.get("V_off")

        Function.__init__(self, [layer], [])

    def _scale_value(self):
        return amp_ratio(self._current_amp, self._voltage_amp, self._scale_exponent)

    def _saturation_current(self):
        return self._I_s_base * self._scale_value()



    def eval(self):
        """Energy term of the non-linearity.

        Returns:
        """
        v = self._layer.state
        abs_v = torch.abs(v)
        Is = tensor_like(self._saturation_current(), v)
        vt = torch.as_tensor(self.vt, dtype=v.dtype, device=v.device)
        voff = torch.as_tensor(self.v_off, dtype=v.dtype, device=v.device)
        excess = torch.clamp(abs_v - voff, min=0.0)


        energy = Is * vt * (torch.exp((excess) / vt) - torch.exp(-voff / vt)) - Is * excess

        return energy.flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        """Returns the gradient function wrt the given layer."""
        dictionary = {self._layer: self._grad_layer}
        return dictionary[layer]

    def _grad_layer(self):
        """Analytical gradient of the exponential diode energy wrt the layer state.
        I think I need to multiply it by 1/current amplification because essentially at the node after the amplification"""
        v = self._layer.state
        v_eff = v
        abs_v = torch.abs(v_eff)
        Is = tensor_like(self._saturation_current(), v)
        vt = torch.as_tensor(self.vt, dtype=v.dtype, device=v.device)
        voff = torch.as_tensor(self.v_off, dtype=v.dtype, device=v.device)

        grad_abs = Is * (torch.exp((abs_v - voff) / vt) - 1.0)
        grad = grad_abs * torch.sign(v)

        return grad


    def a_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)

    def b_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)


class SingleExponentialNonLinearInteraction(Function):
    """Interaction with the single exponential diode non-linearity (split forward/reverse).

    The layer is split into forward- and reverse-oriented diodes by index:
    first half forward, second half reverse.
    """

    def __init__(self, layer, params, voltage_amp, current_amp):
        self._layer = layer
        index = layer_index(layer)
        self._scale_exponent = index - 1
        self._I_s_base = params.get("I_s")
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self.I_s = self._I_s_base * self._scale_value()
        self.vt = params.get("V_t")
        self.v_off = params.get("V_off")

        Function.__init__(self, [layer], [])

    def _scale_value(self):
        return amp_ratio(self._current_amp, self._voltage_amp, self._scale_exponent)

    def _saturation_current(self):
        return self._I_s_base * self._scale_value()

    def _split_mask(self, v_flat):
        features = v_flat.shape[1]
        reverse_mask_nodes = torch.arange(features, device=v_flat.device) >= (features // 2)
        return reverse_mask_nodes.unsqueeze(0).expand_as(v_flat)

    def eval(self):
        """Energy term of the non-linearity."""
        v = self._layer.state
        v_flat = v.reshape(v.shape[0], -1)
        Is = tensor_like(self._saturation_current(), v)
        vt = torch.as_tensor(self.vt, dtype=v.dtype, device=v.device)
        voff = torch.as_tensor(self.v_off, dtype=v.dtype, device=v.device)

        excess_fwd = torch.clamp(v_flat - voff, min=0.0)
        excess_rev = torch.clamp(-v_flat - voff, min=0.0)

        energy_fwd = Is * vt * (torch.exp(excess_fwd / vt) - torch.exp(-voff / vt)) - Is * excess_fwd
        energy_rev = Is * vt * (torch.exp(excess_rev / vt) - torch.exp(-voff / vt)) - Is * excess_rev

        reverse_mask = self._split_mask(v_flat)
        energy = torch.where(reverse_mask, energy_rev, energy_fwd)
        return energy.sum(dim=1)

    def grad_layer_fn(self, layer):
        """Returns the gradient function wrt the given layer."""
        if layer is not self._layer:
            raise ValueError("Requested gradient for an unmanaged layer.")
        return self._grad_layer

    def _grad_layer(self):
        v = self._layer.state
        v_flat = v.reshape(v.shape[0], -1)
        Is = tensor_like(self._saturation_current(), v)
        vt = torch.as_tensor(self.vt, dtype=v.dtype, device=v.device)
        voff = torch.as_tensor(self.v_off, dtype=v.dtype, device=v.device)

        excess_fwd = torch.clamp(v_flat - voff, min=0.0)
        excess_rev = torch.clamp(-v_flat - voff, min=0.0)

        grad_fwd = Is * (torch.exp(excess_fwd / vt) - 1.0)
        grad_rev = -Is * (torch.exp(excess_rev / vt) - 1.0)

        reverse_mask = self._split_mask(v_flat)
        grad = torch.where(reverse_mask, grad_rev, grad_fwd)
        return grad.reshape_as(v)

    def a_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)

    def b_coef_fn(self, layer):
        return lambda: torch.zeros_like(self._layer.state)


class NudgingInteraction(LFunction):
    """Interaction of the nudging force

    A nudging interaction is defined between a layer and the force acting on it

    use_constant_force (bool): whether nudging is achieved using a constant force F = beta * dC(s_free)/ds, or a varying force F = beta * dC(s)/ds

    Attributes
    ----------
    _layer (Layer): the layer involved in the interaction
    _force: the force acting on the layer
    """

    def __init__(self, layer, force):
        """Creates an instance of NudgingInteraction

        Args:
            layer (Layer): the layer involved in the interaction
            force: the force acting on the layer
        """

        self._layer = layer
        self._force = force

        LFunction.__init__(self, [layer], [])

    def eval(self):
        """Energy function of the nudging interaction

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        return - self._layer.state.mul(self._force.state).flatten(start_dim=1).sum(dim=1)

    def grad_layer_fn(self, layer):
        """Returns the function that computes the coefficient b for a given layer"""
        dictionary = {self._layer: self._b_coef_layer}
        return dictionary[layer]

    def _b_coef_layer(self):
        """Returns the linear influence of the bias on the layer's state.

        Returns:
            Tensor of shape (batch_size, layer_shape) and type float32: the linear contribution
        """

        return - self._force.state

    '''def set_output_force(self, force):
        """Set the force of output nodes (used to set the output error values in equilibrium propagation)

        Args:
            force (Tensor): tensor of shape (batch_size, output_size). Type is float32.
        """

        # TODO: check that the shape of the force tensor is the same as the output layer

        self._output_force.state = force.to(self._device)

    def reset_output_force(self):
        """Set the force of output nodes to zero"""
        self._output_force.state = torch.zeros_like(self._output_layer.state)'''


class SumSeparableFunction(Function):
    """
    Class to build a sum-separable function from individual functions.

    Attributes
    ----------
    functions (list of Function): list of all functions

    Methods
    -------
    eval()
        Returns the value of the sum-separable function (for the current configuration)
    """

    def __init__(self, layers, params, interactions):
        """Creates an instance of SumSeparableFunction."""

        Function.__init__(self, layers, params)
        self._interactions = interactions

    def eval(self):
        """Returns the value of the function for the current configuration.

        Returns:
            Tensor of shape (batch_size,) and type float32. Vector of values for each of the examples in the current mini-batch
        """

        return sum([interaction.eval() for interaction in self._interactions])

    def grad_layer_fn(self, layer):
        """Returns the gradient function wrt the given layer

        Overrides the default method of Function.
        The default method of Function computes the gradient of the sum of the individual functions.
        The new method computes the sum of the gradients of the individual functions.

        Args:
            layer (Layer): the layer whose gradient wrt the function we want to compute

        Returns:
            function that computes the gradient of the corresponding layer
        """

        fns = [interaction.grad_layer_fn(layer) for interaction in self._interactions if layer in interaction.layers()]

        return lambda: sum([fn() for fn in fns])

    def grad_param_fn(self, param):
        """Returns the gradient function wrt the given parameter

        Overrides the default method of Function.
        The default method of Function computes the gradient of the sum of the individual functions.
        The new method computes the sum of the gradients of the individual functions.

        Args:
            param (Parameter): the param whose gradient wrt the function we want to compute

        Returns:
            function that computes the gradient of the corresponding param
        """

        fns = [interaction.grad_param_fn(param) for interaction in self._interactions if param in interaction.params()]

        return lambda: sum([fn() for fn in fns])

    def second_fn(self, param):
        """Returns the function: direction -> d^2E / dtheta ds * direction, wrt the parameter (theta)

        Default implementation, valid for any parameter, any layers, and any function

        Returns:
            functions that takes in a dictionary of Tensors of shape (batch_size, layer_shape), and returns a Tensor of shape param_shape
        """

        fns = [interaction.second_fn(param) for interaction in self._interactions if param in interaction.params()]

        return lambda direction: sum([fn(direction) for fn in fns])

    def a_coef_fn(self, layer):
        """Returns a function that computes the coefficient a wrt the given layer

        # FIXME: not all instance of Function have a a_coef_fn method

        Args:
            layer (Layer): the layer whose coefficient a we want to compute

        Returns:
            function that computes the coefficient a of the corresponding layer
        """

        fns = [interaction.a_coef_fn(layer) for interaction in self._interactions if layer in interaction.layers()]
        fns = [fn for fn in fns if fn]

        return lambda: sum([fn() for fn in fns])

    def b_coef_fn(self, layer):
        """Returns a function that computes the coefficient b wrt the given layer

        # FIXME: not all instance of Function have a b_coef_fn method

        Args:
            layer (Layer): the layer whose coefficient b we want to compute

        Returns:
            function that computes the coefficient b of the corresponding layer
        """

        fns = [interaction.b_coef_fn(layer) for interaction in self._interactions if layer in interaction.layers()]

        return lambda: sum([fn() for fn in fns])

    def a_coef_components(self, layer):
        """Return individual a-coefficient contributions for a layer."""
        components = []
        for idx, interaction in enumerate(self._interactions):
            if layer in interaction.layers():
                a_fn = getattr(interaction, "a_coef_fn", None)
                if a_fn is None:
                    continue
                contrib = a_fn(layer)()
                components.append({
                    "interaction": f"{interaction.__class__.__name__}[{idx}]",
                    "value": contrib,
                })
        return components

    def b_coef_components(self, layer):
        """Return individual b-coefficient contributions for a layer.

        Useful for debugging: instead of only seeing the summed b(z),
        this exposes each interaction's contribution for the given layer.
        """
        components = []
        for idx, interaction in enumerate(self._interactions):
            if layer in interaction.layers():
                b_fn = getattr(interaction, "b_coef_fn", None)
                if b_fn is None:
                    continue
                contrib = b_fn(layer)()
                components.append({
                    "interaction": f"{interaction.__class__.__name__}[{idx}]",
                    "value": contrib,
                })
        return components
