import torch
from torch.nn.modules import activation

from model.variable.layer import Layer, InputLayer, LinearLayer



class ResistiveInputLayer(InputLayer):
    """
    Class used to implement an input layer of a resistive network

    Attributes
    ----------
    gain (float): the 'gain' (or 'amplification factor') by which input variables are multiplied
    """

    def __init__(self, shape, gain, batch_size=1, device=None):
        """Creates an instance of ResistiveInputLayer

        Args:
            shape (tuple of int): shape of the tensor used to represent the state of the layer
            gain (float32): amplification factor by which input variables are multiplied
            batch_size (int, optional): the size of the current batch processed. Default: 1
            device (str, optional): the device on which to run the layer's tensor. Either `cuda' or `cpu'. Default: None
        """

        InputLayer.__init__(self, shape, batch_size=batch_size, device=device)

        self._gain = gain
        self._last_input_values = None
        self._last_input_mode = None

    def set_input(self, input_values, mode = "train"):
        """Set the input values

        We duplicate the inputs and invert one set (method used to overcome the constraint of non-negative weights in resistive networks)
        We then multiply the inputs by the gain (amplification factor)

        Args:
            input_values (Tensor): input values
        """
        self._last_input_values = input_values
        self._last_input_mode = mode
        if mode == "train":
            self._state = self._gain * torch.cat((input_values, -input_values), 1)
        elif mode == "debug":
            self._state = self._gain * torch.cat((input_values, torch.zeros_like(input_values)), 1)

    def refresh_input(self):
        if self._last_input_values is None:
            return
        self.set_input(self._last_input_values, mode=self._last_input_mode or "train")




class NonlinearResistiveLayer(Layer):
    """
    Class used to implement a nonlinear resistive layer (a resistive layer with diodes to implement nonlinearities)

    Methods
    -------
    activate():
        Returns the value of the variable's state, clamped between 0 and +infinity for excitatory units, and between -infinity and 0 for inhibitory units
    """

    def __init__(self, shape, batch_size = 1, device = None, non_linearity = None):
        """Initialize NonlinearResistiveLayer

        Args:
            shape (tuple): shape of the layer
            batch_size (int, optional): the size of the current batch processed. Default: 1
            device (str, optional): device to use. Default: None
            activation_mode (str, optional): 'perfect_diode' for hard clamping, 'hard_sigmoid' for no clamping. Default: 'perfect_diode'
        """
        super().__init__(shape, batch_size=batch_size, device=device)   # parent constructor
        self.non_linearity = non_linearity

    def activate(self):
        """Returns the value of the layer's state

        For 'perfect_diode': clamped between 0 and +infinity for excitatory units, and between -infinity and 0 for inhibitory units
        For 'hard_sigmoid': no clamping applied (soft constraints handled in pre_activate)
        For 'linear': no clamping applied (linear activation)
        """

        if self.non_linearity == 'perfect_diode':

            activation_mode = 'hard_clip'

        elif (
            self.non_linearity == 'hard_sigmoid'
            or self.non_linearity == 'linear'
            or self.non_linearity == 'lpw_diode'
            or self.non_linearity == 'double_diode_quadratic'
            or self.non_linearity == 'double_diode_exponential'
            or self.non_linearity == 'single_diode_exponential'
            or self.non_linearity == 'experimental'
        ):
            activation_mode = 'no_clip'
        else:
            raise ValueError(f"Unknown non-linearity: {self.non_linearity}")


        dimension = self._shape[0] // 2  # number of excitatory units = number of inhibitory units = number of units / 2

        if activation_mode == 'hard_clip':
            # Apply hard clamping (original behavior)
            excitatory = self._state[:,:dimension].clamp(min=0., max=None)  # the first half of the units are excitatory units
            inhibitory = self._state[:,dimension:].clamp(min=None, max=0.)  # the second half of the units are inhibitory units
        elif activation_mode == "no_clip":
            # No clamping - soft constraints handled in pre_activate (hard_sigmoid) or linear activation (linear)
            excitatory = self._state[:,:dimension]
            inhibitory = self._state[:,dimension:]

        return torch.cat((excitatory, inhibitory), 1)  # we concatenate excitatory and inhibitory units


class ConvLayer(NonlinearResistiveLayer):
    """Convolutional layer with the non-linearity applied"""

    def __init__(self, shape, batch_size = 1, device=None, non_linearity = None):
        super().__init__(shape, batch_size=batch_size, device=device, non_linearity = non_linearity)


class PoolLayer(LinearLayer):
    """Simple linear layer used for the poooling stage."""

    def __init__(self, shape, batch_size = 1, device=None):
        super().__init__(shape, batch_size=batch_size, device=device)
