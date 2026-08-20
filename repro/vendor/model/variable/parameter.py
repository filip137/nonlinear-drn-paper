from abc import ABC, abstractmethod
import numpy as np
from sympy.core.numbers import Infinity
import torch

from model.variable.variable import Variable



class Parameter(Variable, ABC):
    """Abstract class for parameter variables

    Attributes
    ----------
    _non_negative (bool): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False)

    Methods
    -------
    get()):
        Returns the state of the parameter
    clamp_():
        Clamps the parameter's state in its range of permissible values (in place operation)
    """

    def __init__(self, shape, device, non_negative = True, min_cond = None, max_cond = None):
        """Initializes an instance of Parameter

        Args:
            shape (tuple of ints): Shape of the tensor used to represent the parameter. Type is float32.
            device (str): Either 'cpu' or 'cuda'.
            non_negative (bool, optional): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False). Default: False
            min_cond (float, optional): Lower clamp bound for the parameter's state. Defaults to 0 for non-negative parameters, otherwise no clamp.
            max_cond (float, optional): Upper clamp bound for the parameter's state. Defaults to inf for non-negative parameters, otherwise no clamp.
        """

        Variable.__init__(self, shape)

        self._state = torch.empty(*shape, dtype=torch.float32, device=device)
        self._non_negative = non_negative

        self.min_cond = min_cond
        self.max_cond = max_cond

    def get(self):
        """Returns the state of the parameter"""
        return self._state

    def clamp_(self):
        """Clamps the parameter's state in its range of permissible values (in place operation)"""
        clamp_min = self.min_cond
        clamp_max = self.max_cond

        if self._non_negative:
            if clamp_min is None:
                clamp_min = 0.0
            if clamp_max is None:
                clamp_max = float('inf')

        if clamp_min is not None or clamp_max is not None:
            self._state.clamp_(min=clamp_min, max=clamp_max)


class Bias(Parameter):
    """Class for biases

    Methods
    -------
    init_state(gain):
        Initializes the bias tensor
    """

    _counter = 0

    def __init__(self, shape, gain, device):
        """Initializes an instance of Bias

        Args:
            shape (tuple of ints): Shape of the bias Tensor. Type is float32.
        """

        Parameter.__init__(self, shape, device=device)

        self.init_state(gain)

        self.name = 'Bias_{}'.format(Bias._counter)

        Bias._counter += 1

    def init_state(self, gain):
        """Initializes the bias tensor to zero, i.e. b=0."""

        # TODO: implement recommended initialization schemes for biases, instead of zero

        # torch.nn.init.constant_(self._state, 0.)
        torch.nn.init.uniform_(self._state, -gain, +gain)


class DenseWeight(Parameter):
    """Class for dense ('fully connected') weights

    Methods
    -------
    init_state(gain, mode):
        Initializes the weight tensor
    """

    _counter = 0

    def __init__(self, layer_pre_shape, layer_post_shape, gain, device, clamp=False,
                 clamp_min=None, clamp_max=None, init_mode='kaiming_uniform'):
        """Initializes an instance of DenseWeight

        Args:
            layer_pre_shape (tuple of ints): shape of the pre-synaptic layer
            layer_post_shape (tuple of ints): shape of the post-synaptic layer
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            clamp (bool, optional): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False). Default: False
            clamp_min (float, optional): Lower clamp bound applied when clamp=True. Defaults to 0.
            clamp_max (float, optional): Upper clamp bound applied when clamp=True. Defaults to 100e-6.
        """

        shape = layer_pre_shape + layer_post_shape
        Parameter.__init__(self, shape, device=device, non_negative=clamp,
                           min_cond=clamp_min, max_cond=clamp_max)

        self._layer_pre_shape = layer_pre_shape
        self._layer_post_shape = layer_post_shape

        self.init_state(gain, mode=init_mode)
        self.clamp_()

        self.name = 'DenseWeight_{}'.format(DenseWeight._counter)

        DenseWeight._counter += 1

    def init_state(self, gain, mode='kaiming_uniform'):
        """Initializes the weight tensor according to a uniform or normal distribution.
        Args:
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            mode (str, optional): method to initialize the weight tensor. Either 'xavier_uniform', 'xavier_normal', 'kaiming_uniform' or 'kaiming_normal'. Default: 'xavier_uniform'.
        """

        size_pre = 1
        for dim in self._layer_pre_shape: size_pre *= dim
        size_post = 1
        for dim in self._layer_post_shape: size_post *= dim

        if mode == 'xavier_uniform':
            # half xavier uniform
            scale = gain * 0.5 * np.sqrt(6. / (size_pre + size_post))
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif mode == 'xavier_normal':
            # half xavier normal
            scale = gain * 0.5 * np.sqrt(2. / (size_pre + size_post))
            torch.nn.init.normal_(self._state, std=scale)
        elif mode == 'kaiming_uniform':
            # half kaiming uniform
            # scale = gain * 0.5 * np.sqrt(3. / size_pre)
            scale = gain * np.sqrt(1. / size_pre)
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif mode == 'bounded_uniform':
            # direct range [0, gain]
            torch.nn.init.uniform_(self._state, 0.0, self.max_cond)
        elif mode == 'Kendall':
            lower = 1e-7
            upper = 0.08 / np.sqrt(size_pre + size_post)
            torch.nn.init.uniform_(self._state, lower, upper)
        else:  #  mode == 'kaiming_normal'
            # half kaiming normal
            scale = gain * 0.5 * np.sqrt(1. / size_pre)
            torch.nn.init.normal_(self._state, std=scale)



class ConvWeight(Parameter):
    """Class for convolutional weights

    Methods
    -------
    init_state(gain, mode):
        Initializes the convolutional weight tensor
    """

    _counter = 0

    def __init__(self, shape, gain, device, clamp,
                 clamp_min=0, clamp_max=Infinity, init_mode='kaiming_uniform'):
        """Initializes an instance of ConvWeight

        Args:
            shape (tuple of ints): shape of the convolutional weight tensor. Shape is (out_channels, in_channels, height, width).
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            clamp (bool, optional): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False). Default: False
            init_mode (str, optional): initialization mode ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'). Default: 'kaiming_uniform'.
        """

        Parameter.__init__(self, shape, device=device, non_negative=clamp,
                           min_cond=clamp_min, max_cond=clamp_max)

        self.init_state(gain, init_mode=init_mode)
        self.clamp_()

        self.name = 'ConvWeight_{} '.format(ConvWeight._counter)
        ConvWeight._counter += 1

    def init_state(self, gain, init_mode='kaiming_uniform'):
        """Initializes the weight tensor.

        Args:
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            mode (str, optional): method to initialize the weight tensor. Either 'xavier_uniform', 'xavier_normal', 'kaiming_uniform' or 'kaiming_normal'. Default: 'kaiming_normal'.
        """

        (channels_out, channels_in, width, height) = self._shape
        size_pre = channels_in * width * height
        size_post = channels_out

        if init_mode == 'xavier_uniform':
            # half xavier uniform
            scale = gain * 0.5 * np.sqrt(6. / (size_pre + size_post))
            torch.nn.init.uniform_(self._state, -scale, +scale)
        elif init_mode == 'xavier_normal':
            # half xavier normal
            scale = gain * 0.5 * np.sqrt(2. / (size_pre + size_post))
            torch.nn.init.normal_(self._state, std=scale)
        elif init_mode == 'kaiming_uniform':
            # half kaiming uniform
            # scale = gain * 0.5 * np.sqrt(3. / size_pre)
            scale = gain * np.sqrt(1. / size_pre)
            torch.nn.init.uniform_(self._state, -scale, +scale)
        else:  #  mode == 'kaiming_normal'
            # half kaiming normal
            scale = gain * 0.5 * np.sqrt(1. / size_pre)
            torch.nn.init.normal_(self._state, std=scale)


class PoolWeight(Parameter):
    """Class for pooling (resistive) weights

    Methods
    -------
    init_state(gain, mode):
        Initializes the convolutional weight tensor
    """

    _counter = 0

    def __init__(self, shape, gain, device, clamp,
                 clamp_min=0, clamp_max=1):
        """Initializes an instance of ConvWeight

        Args:
            shape (tuple of ints): shape of the convolutional weight tensor. Shape is (out_channels, in_channels, height, width).
            gain (float32): Number used to scale the weight tensor (~ proportional to the standard deviations of the weight)
            clamp (bool, optional): whether the range of permissible values for the parameter's state is [0,infty] (True) or [-infty, +infty] (False). Default: False
        """

        Parameter.__init__(self, shape, device=device, non_negative=clamp,
                           min_cond=clamp_min, max_cond=clamp_max)


        self.gain = gain
        self.init_state()
        self.clamp_()
        self.name = 'PoolWeight_{} '.format(PoolWeight._counter)
        PoolWeight._counter += 1

    def init_state(self):
        """Initializes the weight tensor.

        Args:
            mode (str, optional): method to initialize the weight tensor. Either 'xavier_uniform', 'xavier_normal', 'kaiming_uniform' or 'kaiming_normal'. Default: 'kaiming_normal'.
        """

        (channels_out, channels_in, width, height) = self._shape
        size_pre = channels_in * width * height

        # Match the conv kaiming_normal scale (same as your ConvWeight 'else' branch)
        w0 = self.gain * 1 * np.sqrt(1.0 / size_pre)

        # Just set the constant; Parameter.clamp_() will handle min/max
        torch.nn.init.constant_(self._state, w0)
        self._state.requires_grad = False


class AmplificationParameter(Parameter):
    """Trainable positive scalar used for DRN amplification factors."""

    _MIN_VALUE = 1e-12

    def __init__(self, name, value, device, min_value=None, max_value=None):
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {value}.")
        if min_value is None:
            min_cond = self._MIN_VALUE
        else:
            min_cond = max(float(min_value), self._MIN_VALUE)
        max_cond = None if max_value is None else float(max_value)
        if max_cond is not None and max_cond <= min_cond:
            raise ValueError(f"{name} max bound must be greater than min bound.")
        if value < min_cond:
            raise ValueError(f"{name} initial value {value} is below lower bound {min_cond}.")
        if max_cond is not None and value > max_cond:
            raise ValueError(f"{name} initial value {value} is above upper bound {max_cond}.")

        Parameter.__init__(
            self,
            (1,),
            device=device,
            non_negative=True,
            min_cond=min_cond,
            max_cond=max_cond,
        )
        torch.nn.init.constant_(self._state, value)
        self.name = name

    def init_state(self):
        pass
