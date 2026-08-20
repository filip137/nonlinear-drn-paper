from tokenize import Double
import torch
from torch.nn import Hardsigmoid

from model.function.interaction import HardSigmoidNonLinearInteraction, SumSeparableFunction
from model.resistive.layer import PoolLayer, ResistiveInputLayer, NonlinearResistiveLayer, ConvLayer
from model.variable.layer import LinearLayer, layer_index
from model.variable.parameter import AmplificationParameter, Bias, DenseWeight, ConvWeight, PoolWeight
# from model.resistive.parameter import TiedDenseWeight as DenseWeight
from model.function.interaction import build_bias_interaction
from model.resistive.interaction import BasePoolResistive, AveragePoolResistive, MaxPoolResistive, DenseResistive, SignedDenseResistive, ConvResistive
from model.function.interaction import (
    DoubleQuadraticNonLinearInteraction,
    DoubleExponentialNonLinearInteraction,
    SingleExponentialNonLinearInteraction,
    LpwNonLinearInteraction,
)
POOLING_INTERACTIONS = {
    "max": MaxPoolResistive,
    "avg": AveragePoolResistive,
}



class DeepResistiveEnergy(SumSeparableFunction):
    """Energy function (power dissipation) of a deep resistive network (DRN)

    The model consists of multiple layers, Successive layers are densely connected.
    """

    def __init__(self, layer_shapes, weight_gains, input_gain,
                 non_linearity, exponential_diode_param, quadratic_diode_param, hard_sigmoid_param,
                 voltage_amp, current_amp,
                 weight_min=None, weight_max=None,
                 weight_init_mode='kaiming_uniform', conv_pipeline=None,
                 pooling_mode="avg", bias_scale_mode="legacy",
                 bias_interaction_type="linear", signed_weights=False,
                 learn_voltage_amp=False, learn_current_amp=False,
                 learn_input_gain=False,
                 voltage_amp_min=None, voltage_amp_max=None,
                 current_amp_min=None, current_amp_max=None,
                 input_gain_min=None, input_gain_max=None):
        """Creates an instance of a dense Hopfield network

        Args:
            layer_shapes (list of tuple of ints): the shapes of the tensors representing the layers of the network
            weight_gains (list of float32): the gains of the weights used at initialization
            input_gain (float): the gain of input variables (input voltage sources)
            non_linearity (str): type of non-linearity ('perfect_diode', 'lpw_diode', 'hard_sigmoid', 'linear')
            voltage_amp (float): voltage amplification factor.
            current_amp (float): current amplification factor.
            weight_min (float, optional): Lower clamp bound applied to the conductance weights.
            weight_max (float, optional): Upper clamp bound applied to the conductance weights.
        """

        self._input_gain_fixed = float(input_gain)
        self._input_gain_param = (
            AmplificationParameter("InputGain", input_gain, device=None, min_value=input_gain_min, max_value=input_gain_max)
            if learn_input_gain
            else None
        )
        self._input_amplifier = self._input_gain_source()
        self._voltage_amp_fixed = float(voltage_amp)
        self._current_amp_fixed = float(current_amp)
        self._voltage_amp_param = (
            AmplificationParameter("VoltageAmp", voltage_amp, device=None, min_value=voltage_amp_min, max_value=voltage_amp_max)
            if learn_voltage_amp
            else None
        )
        self._current_amp_param = (
            AmplificationParameter("CurrentAmp", current_amp, device=None, min_value=current_amp_min, max_value=current_amp_max)
            if learn_current_amp
            else None
        )
        self._amplification_params = [
            param
            for param in (self._voltage_amp_param, self._current_amp_param, self._input_gain_param)
            if param is not None
        ]
        self._voltage_amp = self._voltage_amp_source()
        self._current_amp = self._current_amp_source()
        self._non_linearity = non_linearity
        self._bias_scale_mode = bias_scale_mode
        self._bias_interaction_type = bias_interaction_type
        self._signed_weights = signed_weights
        # Store diode parameter dictionaries so downstream utilities (e.g., Monitor)
        # can introspect saturation bounds without threading them through manually.
        self._quadratic_diode_param = dict(quadratic_diode_param or {})
        self._exponential_diode_param = dict(exponential_diode_param or {})
        self._hardsigmoid_params = dict(hard_sigmoid_param or {})
        self._layer_shapes = layer_shapes
        self._weight_gains = weight_gains
        self._weight_min = weight_min
        self._weight_max = weight_max
        self._weight_init_mode = weight_init_mode
        self._conv_pipeline = list(conv_pipeline or [])
        has_pooling_stage = any(conf.get("mode") == "pooling" for conf in self._conv_pipeline)
        self._pooling_mode = pooling_mode if has_pooling_stage else None
        if has_pooling_stage and self._pooling_mode is None:
            self._pooling_mode = "avg"



        num_conv_stages = len(self._conv_pipeline)
        if num_conv_stages and len(layer_shapes) < num_conv_stages + 2:
            raise ValueError(
                "layer_shapes must specify input, conv stages, and at least one dense/output layer."
            )

        input_shape = layer_shapes[0]
        conv_stage_shapes = layer_shapes[1 : 1 + num_conv_stages]
        hidden_shapes = layer_shapes[1 + num_conv_stages : -1]
        output_shape = layer_shapes[-1]

        input_layer = ResistiveInputLayer(input_shape, gain=self._input_gain_source(), device=None)  # input layer
        self._input_layer = input_layer

        convpool_layers = []
        convpool_modes = []
        PoolInteraction = None
        if self._conv_pipeline:
            if has_pooling_stage and self._pooling_mode not in POOLING_INTERACTIONS:
                raise ValueError(f"Unknown pooling_mode '{self._pooling_mode}', expected one of {tuple(POOLING_INTERACTIONS.keys())}")
            if has_pooling_stage:
                PoolInteraction = POOLING_INTERACTIONS[self._pooling_mode]
            for conf, shape in zip(self._conv_pipeline, conv_stage_shapes):
                mode = conf.get("mode")
                if mode == "convolution":
                    convpool_layer = ConvLayer(shape, device=None, non_linearity=non_linearity)
                elif mode == "pooling":
                    convpool_layer = PoolLayer(shape, device=None)
                else:
                    raise ValueError(f"Unknown conv_pipeline mode '{mode}'")
                convpool_layers.append(convpool_layer)
                convpool_modes.append(mode)

        hidden_layers = [
            NonlinearResistiveLayer(shape, non_linearity=non_linearity) for shape in hidden_shapes
        ]  # hidden layers
        output_layer = LinearLayer(output_shape, device=None)  # output layer
        layers = [input_layer] + convpool_layers + hidden_layers + [output_layer]
        for depth_index, layer in enumerate(layers):
            layer._depth_index = depth_index



        ### CONV / POOLING PARAMETERS
        conv_specs = []
        if self._conv_pipeline:
            if len(convpool_layers) != len(self._conv_pipeline):
                raise ValueError("conv_pipeline length must match conv stage shapes.")

            prev_layer = input_layer
            for conf, layer in zip(self._conv_pipeline, convpool_layers):
                kernel = tuple(conf["kernel"])
                stride = conf.get("stride")
                padding = conf.get("padding")
                mode = conf.get("mode")
                conv_specs.append(
                    {
                        "pre": prev_layer,
                        "post": layer,
                        "kernel": kernel,
                        "stride": stride,
                        "padding": padding,
                        "mode": mode,
                    }
                )
                prev_layer = layer

        self._validate_conv_shapes(conv_specs) ##maybe I do not really need this?

        conv_weight_gains = weight_gains[: len(conv_specs)]
        convpool_weights = []
        for spec, gain in zip(conv_specs, conv_weight_gains):
            out_channels = spec["post"]._shape[0]
            in_channels = spec["pre"]._shape[0]
            kh, kw = spec["kernel"]
            if spec["mode"] == "convolution":
                convpool_weight = ConvWeight(
                    shape=(out_channels, in_channels, kh, kw),
                    gain=gain,
                    device=None,
                    clamp=True,
                    clamp_min=weight_min,
                    clamp_max=weight_max,
                    init_mode = weight_init_mode
                )
            elif spec["mode"] == "pooling":
                convpool_weight = PoolWeight(
                    shape=(out_channels, in_channels, kh, kw),
                    gain=gain,
                    device=None,
                    clamp=True,
                    clamp_min=weight_min,
                    clamp_max=weight_max,

                )
            convpool_weights.append(convpool_weight)

        convpool_interactions = []
        for spec, convpool_weight in zip(conv_specs, convpool_weights):
            if spec["mode"] == "convolution":
                convpool_interaction = ConvResistive(
                    spec["pre"],
                    spec["post"],
                    convpool_weight,
                    padding=spec["padding"],
                    stride=spec["stride"],
                    dilation=1,
                    voltage_amp=voltage_amp,
                    current_amp=current_amp
                )
            elif spec["mode"] == "pooling":
                if PoolInteraction is None:
                    raise ValueError("pooling_mode must be one of {'max', 'avg'} when conv_pipeline contains pooling stages.")
                convpool_interaction = PoolInteraction(
                    spec["pre"],
                    spec["post"],
                    convpool_weight,
                    stride=spec["stride"],
                    voltage_amp=voltage_amp,
                    current_amp=current_amp
                )
            convpool_interactions.append(convpool_interaction)

        dense_source = convpool_layers[-1] if convpool_layers else input_layer
        downstream_layers = [dense_source] + hidden_layers + [output_layer]
        dense_pairs = list(zip(downstream_layers[:-1], downstream_layers[1:]))
        dense_weight_gains = weight_gains[len(conv_specs):]
        free_layers = [layer for layer, mode in zip(convpool_layers, convpool_modes) if mode != "pooling"] + hidden_layers

        # build the biases
        biases = [Bias(layer._shape, 0., device=None) for layer in free_layers]
        bias_interactions = [
            build_bias_interaction(
                layer,
                bias,
                voltage_amp=self._voltage_amp,
                current_amp=self._current_amp,
                scale_mode=self._bias_scale_mode,
                interaction_type=self._bias_interaction_type,
            )
            for layer, bias in zip(free_layers, biases)
        ]

        # build the weights of the network
        # outs = [True] * (len(edges)-1) + [False]
        if self._signed_weights:
            dense_weights = [
                (
                    DenseWeight(
                        layer_pre.shape, layer_post.shape, gain, device=None, clamp=True,
                        clamp_min=weight_min, clamp_max=weight_max, init_mode=weight_init_mode
                    ),
                    DenseWeight(
                        layer_pre.shape, layer_post.shape, gain, device=None, clamp=True,
                        clamp_min=weight_min, clamp_max=weight_max, init_mode=weight_init_mode
                    ),
                )
                for (layer_pre, layer_post), gain in zip(dense_pairs, dense_weight_gains)
            ]
            dense_params = [weight for pair in dense_weights for weight in pair]
            weight_interactions = [
                SignedDenseResistive(layer_pre, layer_post, weight_pos, weight_neg, self._voltage_amp, self._current_amp)
                for (layer_pre, layer_post), (weight_pos, weight_neg) in zip(dense_pairs, dense_weights)
            ]
        else:
            dense_weights = [
                DenseWeight(
                    layer_pre.shape, layer_post.shape, gain, device=None, clamp=True,
                    clamp_min=weight_min, clamp_max=weight_max, init_mode=weight_init_mode
                )
                for (layer_pre, layer_post), gain in zip(dense_pairs, dense_weight_gains)
            ]
            dense_params = dense_weights
            weight_interactions = [DenseResistive(layer_pre, layer_post, weight, self._voltage_amp, self._current_amp) for (layer_pre, layer_post), weight in zip(dense_pairs, dense_weights)]

        # include nonlinear interactions for all nonlinear layers (conv + hidden)
        non_linear_layers = [layer for layer, mode in zip(convpool_layers, convpool_modes) if mode != "pooling"] + hidden_layers

        if non_linearity == "perfect_diode":
            non_linear_interaction = []

        elif non_linearity == "lpw_diode":
            non_linear_interaction = [
                LpwNonLinearInteraction(
                    layer,
                    quadratic_diode_param,
                    voltage_amp=self._voltage_amp,
                    current_amp=self._current_amp,
                )
                for layer in non_linear_layers
            ]

        elif non_linearity == "hard_sigmoid":
            non_linear_interaction = [HardSigmoidNonLinearInteraction(layer, hard_sigmoid_param, voltage_amp=self._voltage_amp, current_amp = self._current_amp)
                for layer in non_linear_layers
            ]
        elif non_linearity == "double_diode_quadratic":
            non_linear_interaction = [
                DoubleQuadraticNonLinearInteraction(layer, quadratic_diode_param, voltage_amp=self._voltage_amp, current_amp = self._current_amp)
                for layer in non_linear_layers
            ]
        elif non_linearity == "double_diode_exponential":
            non_linear_interaction = [
                DoubleExponentialNonLinearInteraction(layer, exponential_diode_param, voltage_amp=self._voltage_amp, current_amp = self._current_amp)
                for layer in non_linear_layers
            ]
        elif non_linearity == "single_diode_exponential":
            non_linear_interaction = [
                SingleExponentialNonLinearInteraction(
                    layer,
                    exponential_diode_param,
                    voltage_amp=self._voltage_amp,
                    current_amp=self._current_amp,
                )
                for layer in non_linear_layers
            ]
        elif non_linearity == "experimental":
            # The paper repro pack handles the measured I-V curve in the
            # coordinate updater, so there is no analytic energy term here.
            non_linear_interaction = []
        else:
            raise ValueError(
                "Expected non_linearity to be one of the bundled resistive nonlinearities. "
                f"Provided value: {non_linearity!r}."
            )

        # Track all params for device movement, but expose only trainable ones (exclude PoolWeight)
        self._all_params = convpool_weights + dense_params + biases + self._amplification_params
        self._trainable_params = [p for p in self._all_params if not isinstance(p, PoolWeight)]
        interactions = bias_interactions + weight_interactions + non_linear_interaction + convpool_interactions

        # creates an instance of Network; pass all params so set_device moves everything to the right device
        SumSeparableFunction.__init__(self, layers, self._all_params, interactions)
        self._sync_amplification_sources()



    @staticmethod
    def _calc_spatial(shape, kernel_size, stride, padding, dilation=1):
        """Compute convolution output height/width for a single stage."""
        _, h_in, w_in = shape
        kh, kw = kernel_size
        h_out = (h_in + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
        w_out = (w_in + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
        return h_out, w_out

    @classmethod
    def _validate_conv_shapes(cls, conv_specs):
        """Ensure declared layer shapes match the geometry implied by conv_specs."""
        if not conv_specs:
            return
        for spec in conv_specs:
            expected_h, expected_w = cls._calc_spatial(
                spec["pre"]._shape,
                spec["kernel"],
                spec["stride"],
                spec["padding"],
            )
            declared = spec["post"]._shape
            if declared[1] != expected_h or declared[2] != expected_w:
                raise ValueError(
                    f"Layer '{spec['post'].name}' expects spatial "
                    f"{declared[1]}x{declared[2]}, but {spec['pre'].name} "
                    f"with kernel {spec['kernel']}, stride {spec['stride']}, padding {spec['padding']} "
                    f"produces {expected_h}x{expected_w}."
                )



    def __str__(self):
        return 'Deep Resistive Network -- layer shapes={}, weight gains={}, input_gain={}'.format(self._layer_shapes, self._weight_gains, self._input_amplifier)

    # Expose only trainable params (PoolWeight is kept frozen)
    def params(self):
        return self._trainable_params

    def _voltage_amp_source(self):
        return self._voltage_amp_param.state if self._voltage_amp_param is not None else self._voltage_amp_fixed

    def _current_amp_source(self):
        return self._current_amp_param.state if self._current_amp_param is not None else self._current_amp_fixed

    def _input_gain_source(self):
        return self._input_gain_param.state if self._input_gain_param is not None else self._input_gain_fixed

    @property
    def voltage_amp(self):
        return self._voltage_amp_source()

    @property
    def current_amp(self):
        return self._current_amp_source()

    def voltage_amp_value(self):
        source = self._voltage_amp_source()
        if torch.is_tensor(source):
            return float(source.detach().reshape(-1)[0].cpu().item())
        return float(source)

    def current_amp_value(self):
        source = self._current_amp_source()
        if torch.is_tensor(source):
            return float(source.detach().reshape(-1)[0].cpu().item())
        return float(source)

    def input_gain_value(self):
        source = self._input_gain_source()
        if torch.is_tensor(source):
            return float(source.detach().reshape(-1)[0].cpu().item())
        return float(source)

    def amplification_params(self):
        return list(self._amplification_params)

    def amplification_parameters(self):
        return self.amplification_params()

    def named_amplification_params(self):
        for param in self._amplification_params:
            yield param.name, param

    def named_amplification_parameters(self):
        yield from self.named_amplification_params()

    def _sync_amplification_sources(self):
        self._voltage_amp = self._voltage_amp_source()
        self._current_amp = self._current_amp_source()
        self._input_amplifier = self._input_gain_source()
        if hasattr(self, "_input_layer"):
            self._input_layer._gain = self._input_amplifier
        interactions = getattr(self, "_interactions", [])
        for interaction in interactions:
            if hasattr(interaction, "_voltage_amp"):
                interaction._voltage_amp = self._voltage_amp
            if hasattr(interaction, "_current_amp"):
                interaction._current_amp = self._current_amp

    def set_device(self, device):
        super().set_device(device)
        self._sync_amplification_sources()

    def load(self, path):
        super().load(path)
        self._sync_amplification_sources()

    def grad_param_fn(self, param):
        if param is self._input_gain_param:
            def _grad_input_gain():
                was_requires_grad = bool(param.state.requires_grad)
                if not was_requires_grad:
                    param.state.requires_grad = True
                if hasattr(self._input_layer, "refresh_input"):
                    self._input_layer.refresh_input()
                value = torch.mean(self.eval())
                if not value.requires_grad:
                    if not was_requires_grad:
                        param.state.requires_grad = False
                    return torch.zeros_like(param.state)
                grad = torch.autograd.grad(value, param.state, allow_unused=True)[0]
                if not was_requires_grad:
                    param.state.requires_grad = False
                if grad is None:
                    return torch.zeros_like(param.state)
                return grad
            return _grad_input_gain
        if param in self._amplification_params:
            return lambda: self._grad(param, mean=True)
        return super().grad_param_fn(param)
