from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from model.function.interaction import QFunction
from model.variable.layer import LinearLayer, layer_index


def _is_input_layer(layer) -> bool:
    return layer_index(layer) == 0


def _is_first_free_layer(layer) -> bool:
    return layer_index(layer) == 1


class DenseResistive(QFunction):
    """Dense resistive interaction between two layers

    Attributes
    ----------
    _layer_pre (Layer): pre-synaptic layer.
    _layer_post (Layer): post-synaptic layer.
    _weight (DenseWeight): weight tensor between layer_pre and layer_post. Tensor of shape (layer_pre_shape, layer_post_shape). Type is float32.
    """

    def __init__(self, layer_pre, layer_post, dense_weight, voltage_amp, current_amp):
        """Initializes an instance of DenseResistive

        Args:
            layer_pre (Layer): pre-synaptic layer
            layer_post (Layer): post-synaptic layer
            dense_weight (DenseWeight): weight tensor between layer_pre and layer_post. Tensor of shape (layer_pre_shape, layer_post_shape). Type is float32.

        """

        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = dense_weight
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp

        QFunction.__init__(self, [layer_pre, layer_post], [dense_weight])

    def eval(self):
        """Computes the energy term corresponding to this weight tensor.

        Returns:
            Vector of size (batch_size,) and of type float32. Each value is the energy term of an example in the current mini-batch
        """

        layer_pre = self._layer_pre.state.clone()
        if not _is_input_layer(self._layer_pre):
            layer_pre = layer_pre * self._voltage_amp
        layer_post = self._layer_post.state  # / self._layer_post.gain
        layer_post = layer_post
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        for _ in range(dims_post): layer_pre = layer_pre.unsqueeze(-1)  # broadcast layer_pre to (batch_size, shape_pre, shape_post)
        for _ in range(dims_pre): layer_post = layer_post.unsqueeze(1)  # broadcast layer_post to (batch_size, shape_pre, shape_post)
        weight = self._weight.get().unsqueeze(0)  # broadcast weight to (batch_size, shape_pre, shape_post)
        layer_pre_index = layer_index(self._layer_pre)
        return 0.5 * ((layer_pre - layer_post)**2).mul(weight).flatten(start_dim=1).sum(dim=1) * (self._current_amp/self._voltage_amp) ** layer_pre_index
        #return 0.5 * ((layer_pre - layer_post)**2).mul(weight).flatten(start_dim=1).sum(dim=1)

    def a_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer_pre: self._a_coef_layer_pre,
            self._layer_post: self._a_coef_layer_post,
            }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        """Overrides the default implementation of QFunction"""
        dictionary = {
            self._layer_pre: self._b_coef_layer_pre,
            self._layer_post: self._b_coef_layer_post,
            }
        return dictionary[layer]

    def grad_param_fn(self, param):
        """Overrides the default implementation of Function"""
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]

    def _b_coef_layer_pre(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution on layer_pre
        """

        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)  # number of dimensions involved in the tensor product
        weight = self._weight.get()
        dim_weight = len(weight.shape)
        permutation = tuple(range(dims_pre, dim_weight)) + tuple(range(dims_pre))
        b_coef = - torch.tensordot(layer_post, weight.permute(permutation), dims=dims_post)
        b_coef = b_coef * self._current_amp
        return b_coef

    def _a_coef_layer_pre(self):
        """Returns the interaction's linear influence on the pre-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_pre_shape) and type float32: the linear contribution on layer_pre
        """

        dims_pre = len(self._layer_pre.shape)
        a_coef = 0.5 * self._weight.get().flatten(start_dim=dims_pre).sum(dim=-1).unsqueeze(0)

        #if not isinstance(self._layer_post, LinearLayer):
        a_coef = a_coef * self._voltage_amp * self._current_amp

        return a_coef

    def _b_coef_layer_post(self):
        """Returns the interaction's linear influence on the post-synaptic layer.
            this acccounts for the previous layer being amplified -- the inputs are not amplified
        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the linear contribution on layer_post
        """

        layer_pre = self._layer_pre.state
        dims_pre = len(self._layer_pre.shape)  # number of dimensions involved in the tensor product
        b_coef = - torch.tensordot(layer_pre, self._weight.get(), dims=dims_pre)
        if not _is_first_free_layer(self._layer_post):
            b_coef = b_coef * self._voltage_amp
        return b_coef

    def _a_coef_layer_post(self):
        """Returns the interaction's quadratic influence on the post-synaptic layer.

        Returns:
            Tensor of shape (batch_size, layer_post_shape) and type float32: the quadratic contribution on layer_post
        """

        dims = len(self._layer_pre.shape) - 1
        a_coef = 0.5 * self._weight.get().flatten(end_dim=dims).sum(dim=0).unsqueeze(0)

        return a_coef

    def _grad_weight(self):
        """Returns the interaction's gradient wrt the weight

        Returns:
            Tensor of shape weight_shape and type float32: the gradient wrt the weights
        """


        layer_pre = self._layer_pre.state.clone()
        if not _is_input_layer(self._layer_pre):
            layer_pre *= self._voltage_amp
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        for _ in range(dims_post): layer_pre = layer_pre.unsqueeze(-1)
        for _ in range(dims_pre): layer_post = layer_post.unsqueeze(1)
        layer_pre_index = layer_index(self._layer_pre)
        amp = (self._current_amp/self._voltage_amp) ** layer_pre_index
        grad_weight = 0.5 * ((layer_pre - layer_post)**2).mean(dim=0) * amp
        #grad_weight = 0.5 * ((layer_pre - layer_post)**2).mean(dim=0)
        return grad_weight


class SignedDenseResistive(QFunction):
    """Dense resistive interaction with separate positive and negative branches.

    The branch conductances are non-negative. Their sum controls the quadratic
    stabilization, while their difference controls the signed cross coupling.
    """

    def __init__(self, layer_pre, layer_post, positive_weight, negative_weight, voltage_amp, current_amp):
        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight_positive = positive_weight
        self._weight_negative = negative_weight
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp

        QFunction.__init__(self, [layer_pre, layer_post], [positive_weight, negative_weight])

    def _magnitude_weight(self):
        return self._weight_positive.get() + self._weight_negative.get()

    def _signed_weight(self):
        return self._weight_positive.get() - self._weight_negative.get()

    def _expanded_states(self):
        layer_pre = self._layer_pre.state.clone()
        if not _is_input_layer(self._layer_pre):
            layer_pre = layer_pre * self._voltage_amp
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        for _ in range(dims_post):
            layer_pre = layer_pre.unsqueeze(-1)
        for _ in range(dims_pre):
            layer_post = layer_post.unsqueeze(1)
        return layer_pre, layer_post

    def eval(self):
        layer_pre, layer_post = self._expanded_states()
        weight_positive = self._weight_positive.get().unsqueeze(0)
        weight_negative = self._weight_negative.get().unsqueeze(0)
        layer_pre_index = layer_index(self._layer_pre)
        amp = (self._current_amp / self._voltage_amp) ** layer_pre_index

        positive_branch = (layer_pre - layer_post).pow(2).mul(weight_positive)
        negative_branch = (-layer_pre - layer_post).pow(2).mul(weight_negative)
        return 0.5 * (positive_branch + negative_branch).flatten(start_dim=1).sum(dim=1) * amp

    def a_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._a_coef_layer_pre,
            self._layer_post: self._a_coef_layer_post,
        }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._b_coef_layer_pre,
            self._layer_post: self._b_coef_layer_post,
        }
        return dictionary[layer]

    def grad_param_fn(self, param):
        dictionary = {
            self._weight_positive: self._grad_weight_positive,
            self._weight_negative: self._grad_weight_negative,
        }
        return dictionary[param]

    def _b_coef_layer_pre(self):
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.shape)
        weight = self._signed_weight()
        dim_weight = len(weight.shape)
        permutation = tuple(range(dims_pre, dim_weight)) + tuple(range(dims_pre))
        b_coef = -torch.tensordot(layer_post, weight.permute(permutation), dims=dims_post)
        return b_coef * self._current_amp

    def _a_coef_layer_pre(self):
        dims_pre = len(self._layer_pre.shape)
        a_coef = 0.5 * self._magnitude_weight().flatten(start_dim=dims_pre).sum(dim=-1).unsqueeze(0)
        return a_coef * self._voltage_amp * self._current_amp

    def _b_coef_layer_post(self):
        layer_pre = self._layer_pre.state
        dims_pre = len(self._layer_pre.shape)
        b_coef = -torch.tensordot(layer_pre, self._signed_weight(), dims=dims_pre)
        if not _is_first_free_layer(self._layer_post):
            b_coef = b_coef * self._voltage_amp
        return b_coef

    def _a_coef_layer_post(self):
        dims = len(self._layer_pre.shape) - 1
        return 0.5 * self._magnitude_weight().flatten(end_dim=dims).sum(dim=0).unsqueeze(0)

    def _grad_weight_positive(self):
        layer_pre, layer_post = self._expanded_states()
        layer_pre_index = layer_index(self._layer_pre)
        amp = (self._current_amp / self._voltage_amp) ** layer_pre_index
        return 0.5 * (layer_pre - layer_post).pow(2).mean(dim=0) * amp

    def _grad_weight_negative(self):
        layer_pre, layer_post = self._expanded_states()
        layer_pre_index = layer_index(self._layer_pre)
        amp = (self._current_amp / self._voltage_amp) ** layer_pre_index
        return 0.5 * (-layer_pre - layer_post).pow(2).mean(dim=0) * amp


class ConvResistive(QFunction):
    """Convolutional resistive interaction mirroring DenseResistive logic in conv form."""

    def __init__(self, layer_pre, layer_post, conv_weight, padding, stride, dilation, voltage_amp, current_amp):
        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = conv_weight
        QFunction.__init__(self, [layer_pre, layer_post], [conv_weight])
        self._P = padding
        self._S = stride
        self._D = dilation
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp

    def _conv_geometry(self):
        weight = self._weight.get()
        C_out, C_in, Kh, Kw = weight.shape
        x = self._layer_pre.state
        _, _, H_in, W_in = x.shape

        H_out = (H_in + 2 * self._P - self._D * (Kh - 1) - 1) // self._S + 1
        W_out = (W_in + 2 * self._P - self._D * (Kw - 1) - 1) // self._S + 1

        return weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out

    def eval(self, per_sample=True):
        weight, C_out, C_in, Kh, Kw, *_ = self._conv_geometry()

        layer_pre = self._layer_pre.state.clone()
        if not _is_input_layer(self._layer_pre):
            layer_pre = layer_pre * self._voltage_amp
        layer_post = self._layer_post.state  # / self._layer_post.gain


        cols = F.unfold(layer_pre, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        N, C_out, H_out, W_out = layer_post.shape
        K = C_in * Kh * Kw
        L = H_out * W_out

        patches = cols.transpose(1, 2).unsqueeze(2)
        kernels = weight.view(1, 1, C_out, K)
        targets = layer_post.view(N, C_out, L).transpose(1, 2).unsqueeze(-1)

        diff2 = (patches - targets).pow(2)
        weighted = diff2 * kernels
        layer_pre_index = layer_index(self._layer_pre)
        amp = (self._current_amp / self._voltage_amp) ** layer_pre_index
        E_per = 0.5 * weighted.sum(dim=(1, 2, 3)) * amp
        return E_per if per_sample else E_per.sum()

    def im2col(self):
        x = self._layer_pre.state
        N = x.shape[0]
        weight, C_out, _, Kh, Kw, _, _, H_out, W_out = self._conv_geometry()

        cols = F.unfold(x, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        Wflat = weight.view(C_out, -1)
        y = torch.matmul(cols.transpose(1, 2), Wflat.t())
        y = y.transpose(1, 2).reshape(N, C_out, H_out, W_out)
        return y

    def col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._conv_geometry()
        y = self._layer_post.state
        N = y.shape[0]

        L = H_out * W_out
        y_cols = y.reshape(N, C_out, L)
        Wflat = weight.view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)

        x_pre = F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )
        return x_pre

    def a_im2col(self):
        weight, C_out, C_in, Kh, Kw, _, _, H_out, W_out = self._conv_geometry()
        a_per_ch = weight.view(C_out, -1).sum(dim=1).view(1, C_out, 1, 1)
        return a_per_ch.expand(1, C_out, H_out, W_out)

    def a_col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._conv_geometry()
        y = self._layer_post.state
        y_ones = torch.ones_like(y)
        N = y_ones.shape[0]

        L = H_out * W_out
        y_cols = y_ones.reshape(N, C_out, L)
        Wflat = weight.view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)
        x_pre = F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )
        return x_pre

    def a_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._a_coef_layer_pre,
            self._layer_post: self._a_coef_layer_post,
        }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._b_coef_layer_pre,
            self._layer_post: self._b_coef_layer_post,
        }
        return dictionary[layer]

    def _b_coef_layer_pre(self):
        b_coef = -self.col2im()
        b_coef = b_coef * self._current_amp
        return b_coef

    def _b_coef_layer_post(self):
        b_coef = -self.im2col()
        if not _is_first_free_layer(self._layer_post):
            b_coef = b_coef * self._voltage_amp
        return b_coef

    def _a_coef_layer_pre(self):
        a_map = self.a_col2im()
        a_coef = a_map * self._voltage_amp * self._current_amp
        return 0.5 * a_coef

    def _a_coef_layer_post(self):
        a_map = self.a_im2col()
        return 0.5 * a_map

    def _grad_weight(self):
        weight, C_out, C_in, Kh, Kw, *_ = self._conv_geometry()
        x = self._layer_pre.state.clone()
        if not _is_input_layer(self._layer_pre):
            x = x * self._voltage_amp
        y = self._layer_post.state.clone()

        cols = F.unfold(x, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        N, C_out, H_out, W_out = y.shape
        K = C_in * Kh * Kw
        L = H_out * W_out

        patches = cols.transpose(1, 2).unsqueeze(2)
        targets = y.reshape(N, C_out, L).transpose(1, 2).unsqueeze(-1)
        diff2 = (patches - targets).pow(2)
        grad_weight = 0.5 * diff2.sum(dim=1).mean(dim=0)

        layer_pre_index = layer_index(self._layer_pre)
        amp = (self._current_amp / self._voltage_amp) ** layer_pre_index
        return grad_weight.view(C_out, C_in, Kh, Kw) * amp

    def grad_param_fn(self, param):
        dictionary = {self._weight: self._grad_weight}
        return dictionary[param]


class SignedConvResistive(QFunction):
    """Convolutional resistive interaction with signed effective kernels.

    The physical kernels are non-negative branch conductances. Their sum gives
    the quadratic curvature, while their difference gives the signed coupling.
    """

    def __init__(self, layer_pre, layer_post, positive_weight, negative_weight, padding, stride, dilation, voltage_amp, current_amp):
        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight_positive = positive_weight
        self._weight_negative = negative_weight
        QFunction.__init__(self, [layer_pre, layer_post], [positive_weight, negative_weight])
        self._P = padding
        self._S = stride
        self._D = dilation
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp

    def _magnitude_weight(self):
        return self._weight_positive.get() + self._weight_negative.get()

    def _signed_weight(self):
        return self._weight_positive.get() - self._weight_negative.get()

    def _conv_geometry(self):
        weight = self._magnitude_weight()
        C_out, C_in, Kh, Kw = weight.shape
        x = self._layer_pre.state
        _, _, H_in, W_in = x.shape

        H_out = (H_in + 2 * self._P - self._D * (Kh - 1) - 1) // self._S + 1
        W_out = (W_in + 2 * self._P - self._D * (Kw - 1) - 1) // self._S + 1

        return weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out

    def _expanded_states(self):
        weight, C_out, C_in, Kh, Kw, *_ = self._conv_geometry()
        x = self._layer_pre.state.clone()
        if not _is_input_layer(self._layer_pre):
            x = x * self._voltage_amp
        y = self._layer_post.state

        cols = F.unfold(x, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        N, C_out, H_out, W_out = y.shape
        patches = cols.transpose(1, 2).unsqueeze(2)
        targets = y.reshape(N, C_out, H_out * W_out).transpose(1, 2).unsqueeze(-1)
        return patches, targets

    def eval(self, per_sample=True):
        patches, targets = self._expanded_states()
        weight_positive = self._weight_positive.get().view(1, 1, *self._weight_positive.get().shape[:1], -1)
        weight_negative = self._weight_negative.get().view(1, 1, *self._weight_negative.get().shape[:1], -1)
        layer_pre_index = layer_index(self._layer_pre)
        amp = (self._current_amp / self._voltage_amp) ** layer_pre_index

        positive_branch = (patches - targets).pow(2).mul(weight_positive)
        negative_branch = (-patches - targets).pow(2).mul(weight_negative)
        E_per = 0.5 * (positive_branch + negative_branch).sum(dim=(1, 2, 3)) * amp
        return E_per if per_sample else E_per.sum()

    def im2col(self):
        x = self._layer_pre.state
        N = x.shape[0]
        weight, C_out, _, Kh, Kw, _, _, H_out, W_out = self._conv_geometry()

        cols = F.unfold(x, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        Wflat = self._signed_weight().view(C_out, -1)
        y = torch.matmul(cols.transpose(1, 2), Wflat.t())
        y = y.transpose(1, 2).reshape(N, C_out, H_out, W_out)
        return y

    def col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._conv_geometry()
        y = self._layer_post.state
        N = y.shape[0]

        L = H_out * W_out
        y_cols = y.reshape(N, C_out, L)
        Wflat = self._signed_weight().view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)

        return F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )

    def a_im2col(self):
        weight, C_out, C_in, Kh, Kw, _, _, H_out, W_out = self._conv_geometry()
        a_per_ch = weight.view(C_out, -1).sum(dim=1).view(1, C_out, 1, 1)
        return a_per_ch.expand(1, C_out, H_out, W_out)

    def a_col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._conv_geometry()
        y = self._layer_post.state
        y_ones = torch.ones_like(y)
        N = y_ones.shape[0]

        L = H_out * W_out
        y_cols = y_ones.reshape(N, C_out, L)
        Wflat = weight.view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)
        return F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )

    def a_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._a_coef_layer_pre,
            self._layer_post: self._a_coef_layer_post,
        }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._b_coef_layer_pre,
            self._layer_post: self._b_coef_layer_post,
        }
        return dictionary[layer]

    def _b_coef_layer_pre(self):
        return -self.col2im() * self._current_amp

    def _b_coef_layer_post(self):
        b_coef = -self.im2col()
        if not _is_first_free_layer(self._layer_post):
            b_coef = b_coef * self._voltage_amp
        return b_coef

    def _a_coef_layer_pre(self):
        a_map = self.a_col2im()
        return 0.5 * a_map * self._voltage_amp * self._current_amp

    def _a_coef_layer_post(self):
        return 0.5 * self.a_im2col()

    def _grad_weight_positive(self):
        weight, C_out, C_in, Kh, Kw, *_ = self._conv_geometry()
        patches, targets = self._expanded_states()
        grad_weight = 0.5 * (patches - targets).pow(2).sum(dim=1).mean(dim=0)
        layer_pre_index = layer_index(self._layer_pre)
        amp = (self._current_amp / self._voltage_amp) ** layer_pre_index
        return grad_weight.view(C_out, C_in, Kh, Kw) * amp

    def _grad_weight_negative(self):
        weight, C_out, C_in, Kh, Kw, *_ = self._conv_geometry()
        patches, targets = self._expanded_states()
        grad_weight = 0.5 * (-patches - targets).pow(2).sum(dim=1).mean(dim=0)
        layer_pre_index = layer_index(self._layer_pre)
        amp = (self._current_amp / self._voltage_amp) ** layer_pre_index
        return grad_weight.view(C_out, C_in, Kh, Kw) * amp

    def grad_param_fn(self, param):
        dictionary = {
            self._weight_positive: self._grad_weight_positive,
            self._weight_negative: self._grad_weight_negative,
        }
        return dictionary[param]


class BasePoolResistive(QFunction, ABC):
    def __init__(self, layer_pre, layer_post, pool_weight, stride, voltage_amp, current_amp):
        self._layer_pre = layer_pre
        self._layer_post = layer_post
        self._weight = pool_weight
        self._stride = stride
        QFunction.__init__(self, [layer_pre, layer_post], [pool_weight]) ##I am not sure whether to include pool_weight in the parameters
        self._voltage_amp = voltage_amp
        self._current_amp = current_amp
        self._S = stride
        self._P = 0
        self._D = 1


    def _pool_geometry(self):
        weight = self._weight.get()
        C_out, C_in, Kh, Kw = weight.shape
        if C_out != C_in:
            raise ValueError("C_out must equal C_in for pooling")
        x = self._layer_pre.state
        _, _, H_in, W_in = x.shape

        H_out = (H_in + 2 * self._P - self._D * (Kh - 1) - 1) // self._S + 1
        W_out = (W_in + 2 * self._P - self._D * (Kw - 1) - 1) // self._S + 1

        return weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out


    @abstractmethod
    def eval(self, per_sample=True):
        pass


    @abstractmethod
    def im2col(self):
        pass


    @abstractmethod
    def col2im(self):
        pass



    @abstractmethod
    def a_im2col(self):
        pass


    @abstractmethod
    def a_col2im(self):
        pass

    def a_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._a_coef_layer_pre,
            self._layer_post: self._a_coef_layer_post,
        }
        return dictionary[layer]

    def b_coef_fn(self, layer):
        dictionary = {
            self._layer_pre: self._b_coef_layer_pre,
            self._layer_post: self._b_coef_layer_post,
        }
        return dictionary[layer]

    def _b_coef_layer_pre(self):
        b_coef = -self.col2im()
        b_coef = b_coef * self._current_amp
        return b_coef

    def _b_coef_layer_post(self):
        b_coef = -self.im2col()
        if not _is_first_free_layer(self._layer_post):
            b_coef = b_coef * self._voltage_amp
        return b_coef

    def _a_coef_layer_pre(self):
        a_map = self.a_col2im()
        a_coef = a_map * self._voltage_amp * self._current_amp
        return 0.5 * a_coef

    def _a_coef_layer_post(self):
        a_map = self.a_im2col()
        return 0.5 * a_map

    def _grad_weight(self):
        weight, C_out, C_in, Kh, Kw, *_ = self._pool_geometry()
        return torch.zeros_like(weight)

    def grad_param_fn(self, param):
        return {self._weight: self._grad_weight}.get(param, lambda: torch.zeros(1, device=self._weight.state.device))



class AveragePoolResistive(BasePoolResistive):
    def __init__(self, layer_pre, layer_post, pool_weight, stride, voltage_amp, current_amp):
        BasePoolResistive.__init__(self, layer_pre, layer_post, pool_weight, stride, voltage_amp, current_amp)


    def eval(self, per_sample=True):
        weight, C_out, C_in, Kh, Kw, *_ = self._pool_geometry()

        layer_pre = self._layer_pre.state.clone()
        if not _is_input_layer(self._layer_pre):
            layer_pre = layer_pre * self._voltage_amp
        layer_post = self._layer_post.state  # / self._layer_post.gain


        cols = F.unfold(layer_pre, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        N, C_out, H_out, W_out = layer_post.shape
        K = C_in * Kh * Kw
        L = H_out * W_out

        patches = cols.transpose(1, 2).unsqueeze(2)
        kernels = weight.view(1, 1, C_out, K)
        targets = layer_post.view(N, C_out, L).transpose(1, 2).unsqueeze(-1)

        diff2 = (patches - targets).pow(2)
        weighted = diff2 * kernels
        layer_pre_index = layer_index(self._layer_pre)
        amp = (self._current_amp / self._voltage_amp) ** layer_pre_index
        E_per = 0.5 * weighted.sum(dim=(1, 2, 3)) * amp
        return E_per if per_sample else E_per.sum()

    def im2col(self):
        x = self._layer_pre.state
        N = x.shape[0]
        weight, C_out, _, Kh, Kw, _, _, H_out, W_out = self._pool_geometry()

        cols = F.unfold(x, (Kh, Kw), padding=self._P, stride=self._S, dilation=self._D)
        Wflat = weight.view(C_out, -1)
        y = torch.matmul(cols.transpose(1, 2), Wflat.t())
        y = y.transpose(1, 2).reshape(N, C_out, H_out, W_out)
        return y

    def col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        y = self._layer_post.state
        N = y.shape[0]

        L = H_out * W_out
        y_cols = y.reshape(N, C_out, L)
        Wflat = weight.view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)

        x_pre = F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )
        return x_pre

    def a_im2col(self):
        weight, C_out, C_in, Kh, Kw, _, _, H_out, W_out = self._pool_geometry()
        a_per_ch = weight.view(C_out, -1).sum(dim=1).view(1, C_out, 1, 1)
        return a_per_ch.expand(1, C_out, H_out, W_out)

    def a_col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        y = self._layer_post.state
        y_ones = torch.ones_like(y)
        N = y_ones.shape[0]

        L = H_out * W_out
        y_cols = y_ones.reshape(N, C_out, L)
        Wflat = weight.view(C_out, -1)
        cols_pre = torch.matmul(Wflat.t().unsqueeze(0), y_cols)
        x_pre = F.fold(
            cols_pre,
            output_size=(H_in, W_in),
            kernel_size=(Kh, Kw),
            padding=self._P,
            stride=self._S,
            dilation=self._D,
        )
        return x_pre

class MaxPoolResistive(BasePoolResistive):
    def __init__(self, layer_pre, layer_post, pool_weight, stride, voltage_amp, current_amp, max_pooling_mode = "abs_pool"):
        BasePoolResistive.__init__(self, layer_pre, layer_post, pool_weight, stride, voltage_amp, current_amp)
        self.pooling_mode = max_pooling_mode #abs_pool or none
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        self.gain = self._weight.get().mean().item() * (Kh * Kw)
        self.f_idx = None
        self.b_idx = None


    def eval(self, per_sample=True):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()

        # Scale states exactly like BasePoolResistive.eval does
        x = self._layer_pre.state
        if not _is_input_layer(self._layer_pre):
            x = x * self._voltage_amp

        y = self._layer_post.state  # shape (N, C_out, H_out, W_out)

        N = x.shape[0]

        # Winner selection (consistent with your abs_pool routing)
        if self.pooling_mode == "abs_pool":
            _, idx = F.max_pool2d(
                x.abs(),
                kernel_size=(Kh, Kw),
                stride=self._S,
                padding=self._P,
                dilation=self._D,
                return_indices=True,
            )
            # idx shape: (N, C_in, H_out, W_out) since C_out == C_in for pooling
            x_flat = x.view(N, C_in, -1)
            idx_flat = idx.view(N, C_in, -1)
            x_win = torch.gather(x_flat, 2, idx_flat).view(N, C_in, H_out, W_out)
        else:
            # Standard maxpool (signed)
            x_win = F.max_pool2d(
                x,
                kernel_size=(Kh, Kw),
                stride=self._S,
                padding=self._P,
                dilation=self._D,
            )

        # Effective conductance (broadcast safely)
        g = self.gain
        if not torch.is_tensor(g):
            g = torch.tensor(g, device=x.device, dtype=x.dtype)

        if g.ndim == 0:
            g = g.view(1, 1, 1, 1)              # scalar gain
        elif g.ndim == 1:
            g = g.view(1, C_in, 1, 1)            # per-channel gain

        # Energy: 1/2 * g * (x_win - y)^2
        diff2 = (x_win - y).pow(2)
        layer_pre_index = layer_index(self._layer_pre)
        amp = (self._current_amp / self._voltage_amp) ** layer_pre_index
        E_per = 0.5 * (diff2 * g).sum(dim=(1, 2, 3)) * amp

        return E_per if per_sample else E_per.sum()



    def mymax_pool2d(self, x, kernel_size):
        stride = self._S
        padding = self._P
        dilation = self._D
        N = x.shape[0]
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        if self.pooling_mode == "abs_pool":
            _, idx = F.max_pool2d(abs(x), kernel_size = kernel_size, stride = self._S, padding = self._P, dilation = self._D, return_indices = True)
            out = torch.gather(x.view(N, C_in, -1), 2, idx.view(N, C_out, -1))
            values = out.view(N, C_out, H_out, W_out)
            return values, idx

        else:
            values, idx = F.max_pool2d(x, kernel_size = kernel_size, stride = self._S, padding = self._P, dilation = self._D, return_indices = True)
            return values, idx

    def mymax_unpool2d(self, x , y, kernel_size):
        stride = self._S
        padding = self._P
        dilation = self._D
        N = y.shape[0]
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        if self.pooling_mode == "abs_pool":
            _, idx = F.max_pool2d(abs(x), kernel_size = kernel_size, stride = self._S, padding = self._P, dilation = self._D, return_indices = True)
            self.b_idx = idx
            out = torch.gather(x.view(N, C_in, -1), 2, idx.view(N, C_out, -1))
            values = out.view(N, C_out, H_out, W_out)
            x_pre = F.max_unpool2d(input = y, indices = idx, kernel_size = kernel_size, stride=self._S, padding=self._P,output_size=(N, C_out, H_in, W_in))

        else:
            _, idx = F.max_pool2d(x, kernel_size = kernel_size, stride = self._S, padding = self._P, dilation = self._D, return_indices = True)
            x_pre = F.max_unpool2d(input = y, indices = idx, kernel_size = kernel_size, stride=self._S, padding=self._P, output_size=(N, C_out, H_in, W_in))

        if self.f_idx is not None and self.b_idx is not None:
            diff = self.f_idx - self.b_idx
            mismatches = torch.count_nonzero(diff)
            mismatched_norm = mismatches/diff.numel()
            self.f_idx = None
            self.b_idx = None

        return x_pre

    def im2col(self):
        #maybe I can include the gain here
        x = self._layer_pre.state
        weight, _, _, Kh, Kw, *_ = self._pool_geometry()
        y, idx = self.mymax_pool2d(x, kernel_size=(Kh, Kw))
        self.f_idx = idx
        return y  * self.gain
    def col2im(self):
        #maybe I can include the gain here
        x = self._layer_pre.state
        y = self._layer_post.state
        _, _, _, Kh, Kw, _, _, _, _ = self._pool_geometry()
        x_pre = self.mymax_unpool2d(x, y, kernel_size=(Kh, Kw))
        return x_pre * self.gain

    def a_im2col(self):
        #this makes sense if the weight I consider is alwazs self.gain
        #this is independent of the winners
        weight, C_out, C_in, Kh, Kw, _, _, H_out, W_out = self._pool_geometry()
        a_per_ch = torch.ones(1, C_out, H_out, W_out, device = weight.device) * self.gain
        return a_per_ch.expand(1, C_out, H_out, W_out)

    def a_col2im(self):
        weight, C_out, C_in, Kh, Kw, H_in, W_in, H_out, W_out = self._pool_geometry()
        x = self._layer_pre.state
        y = self._layer_post.state
        y_ones = torch.ones_like(y) * self.gain
        N = y_ones.shape[0]

        L = H_out * W_out
        y_cols = y_ones.reshape(N, C_out, L)
        x_mask = self.mymax_unpool2d(x, y_ones, kernel_size=(Kh, Kw))
        return x_mask


    def analytical_grad_layer_fn(self, layer):
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
            return lambda: 2. * a_fn() * layer.state + b_fn()  # tensor of size (batch_size, layer_shape)
