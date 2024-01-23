import typing
from typing import List
from tensorflow.python.keras.utils import tf_utils

from .utils import keras, torch
from typing import Dict


def iid_gaussian(m, d):
    """Generate random values that are I.I.D (independent identically distributed)
    :param m: int
        Hidden Dimensions
    :param d: int
        Depth (half the hidden dimensions)"""
    return torch.randn(size=(m, d))


def orthogonal_gaussian(m: int, d: int):
    """Generate Orthogonal Gaussian distribution's. This is to improve upon MSE (mean squared error)
    inside a performer.
    Args:
        :param m: int
            Hidden Dimensions
        :param d: int
            Depth (half the hidden dimensions)"""

    def orthogonal_square():
        q, _ = torch.linalg.qr(iid_gaussian(d, d))
        return q.t()  # transpose

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    # matrix = tf.concat(blocks, axis=0)
    matrix = torch.vstack(blocks)
    matrix /= torch.sqrt(torch.as_tensor(num_squares + remainder / d))

    return matrix


def softmax_kernel_transformation(data: torch.Tensor,
                                  is_query: bool,
                                  projection_matrix: torch.Tensor = None,
                                  numerical_stabilizer=0.000001):
    """Computes random features for the softmax kernel using FAVOR+ mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    :param data: torch.Tensor
        input data tensor of the shape [B, L, H, D], where: B - batch dimension,
            L - attention dimensions, H - heads, D - depth
    :param is_query: torch.Tensor
        Indicates whether input data is a query oor key tensor
    :param projection_matrix: torch.Tensor
        random Gaussian matrix of shape [M, D], where M stands for the
        number of random features and each D x D sub-block has pairwise orthogonal rows
    :param numerical_stabilizer: float
        small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
    projection_matrix = projection_matrix.type(data.dtype).to(data.device)
    data_normalizer = 1.0 / (
        torch.math.sqrt(torch.math.sqrt(data.size()[-1])))
    data = data_normalizer * data
    ratio = 1.0 / torch.math.sqrt(projection_matrix.size()[0])
    # noinspection SpellCheckingInspection
    # This is the kernel transformation
    data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)
    # Diag Data is masking out the diagonal values. Similar to how the scaled dot product attention works.
    diag_data = torch.square(data)
    dim = data.dim() - 1
    diag_data = diag_data.sum(dim=dim)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=dim)
    last_dims_t = data_dash.dim() - 1
    attention_dims_t = data_dash.dim() - 3
    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - data_dash.max(dim=last_dims_t, keepdim=True)[0])
                + numerical_stabilizer)
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - data_dash.max(dim=last_dims_t, keepdim=True)[0])
                + numerical_stabilizer)

    return data_dash


def relu_kernel_transformation(data: torch.Tensor,
                               projection_matrix: torch.Tensor = None,
                               numerical_stabilizer=0.000001):
    """Computes random features for the ReLU kernel using FAVOR+ mechanism.

    Args:
        :param data: torch.Tensor
            input data tensor of the shape [B, L, H, D], where: B - batch dimension,
            L - attention dimensions, H - heads, D - depth
        :param projection_matrix: torch.Tensor
            random Gaussian matrix of shape [M, D], where M stands for the
            number of random features and each D x D sub-block has pairwise orthogonal rows
        :param numerical_stabilizer: float
            small positive constant for numerical stability.
    """
    projection_matrix = projection_matrix.type(data.dtype).to(data.device)
    m = data.size()[-1]
    m = torch.as_tensor(m).type(data.dtype)
    data_normalizer = 1.0 / torch.math.sqrt(m)
    projection_matmul = torch.einsum("blhd,md->blhm", data, projection_matrix)
    return torch.relu(data_normalizer * projection_matmul + numerical_stabilizer)


def attn_hat(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
             phi_fun=None, random_feats: torch.Tensor = None):
    """
    Args:
        :param query: torch.Tensor
            The Query tensor from the Multi-headed attention mechanism
        :param key: torch.Tensor
        The Key tensor from the Multi-headed attention mechanism
        :param value: torch.Tensor
            The Value tensor from the Multi-headed attention mechanism
        :param phi_fun: Any function
            A function for "phi" If None, default to Softmax kernel transformations
        :param random_feats: torch.Tensor
            The random features for use in phi function in predicting the softmax values
    """
    sequence_length = query.size()[2]
    # B, H, L, D to B, L, H, D
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    if phi_fun is not None:
        q_prime = phi_fun(query, random_feats)
        k_prime = phi_fun(key, random_feats)
    else:
        q_prime = softmax_kernel_transformation(query, projection_matrix=random_feats, is_query=True)  # B L H M
        k_prime = softmax_kernel_transformation(key, projection_matrix=random_feats, is_query=False)  # B L H M

    # B H L D, L B H D
    value = value.permute(2, 0, 1, 3)

    # B L H M, L B H M
    k_prime = k_prime.permute(1, 0, 2, 3)  # L B H M
    q_prime = q_prime.permute(1, 0, 2, 3)  # L B H M

    # noinspection SpellCheckingInspection
    av_attention = torch.einsum("lbhm,lbhd->bhmd", k_prime, value)

    # noinspection SpellCheckingInspection
    av_attention = torch.einsum("lbhm,bhmd->lbhd", q_prime, av_attention)
    # noinspection SpellCheckingInspection
    normalizer = torch.einsum("lbhm,l->bhm", k_prime, torch.ones(sequence_length, dtype=k_prime.dtype,
                                                                 device=k_prime.device))
    # noinspection SpellCheckingInspection
    normalizer = torch.einsum("lbhm,bhm->lbh", q_prime, normalizer)
    av_attention = av_attention.permute(1, 0, 2, 3)  # B L H D
    normalizer = normalizer.permute(1, 0, 2)  # B L H
    normalizer = torch.unsqueeze(normalizer, normalizer.dim())  # B L H 1
    return av_attention / normalizer


def positive_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, random_feats: torch.Tensor):
    """Instead of using ScaledDotProduction, this uses the above Gaussian elements to estimate the answer that
    the full ScaledDotProduction would give.
    Args:
        :param query: torch.Tensor
            The Query tensor from the Multi-headed attention mechanism
        :param key: torch.Tensor
            The Key tensor from the Multi-headed attention mechanism
        :param value:
            The Value tensor from the Multi-headed attention mechanism
        :param random_feats:
            The random features for use in phi function in predicting the softmax values.
        """

    return attn_hat(query, key, value, random_feats=random_feats)


def positive_relu_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, random_feats: torch.Tensor):
    """Instead of using ScaledDotProduction, this uses the above Gaussian elements to estimate the answer that
    the full ScaledDotProduction would give.
    Args:
        :param query: torch.Tensor
            The Query tensor from the Multi-headed attention mechanism
        :param key: torch.Tensor
            The Key tensor from the Multi-headed attention mechanism
        :param value:
            The Value tensor from the Multi-headed attention mechanism
        :param random_feats:
            The random features for use in phi function in predicting the softmax values.
        """

    return attn_hat(query, key, value, random_feats=random_feats, phi_fun=relu_kernel_transformation)


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                 mask: torch.Tensor, name_prefix: str) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        :param query: torch.Tensor
            The Query tensor from the Multi-headed attention mechanism
        :param key: torch.Tensor
            The Key tensor from the Multi-headed attention mechanism
        :param value: torch.Tensor
            The Value tensor from the Multi-headed attention mechanism
        :param mask: torch.Tensor
            For masking out previous outputs
        :param name_prefix: str
            The name prefix for the attention mechanism
    :return: The final tensor object
    """
    matmul_qk = torch.matmul(query, key.permute(0, 1, 3, 2))

    depth = keras.ops.cast(key.size()[-1], query.dtype)
    logits = matmul_qk / torch.math.sqrt(depth)
    logits = keras.ops.cast(logits, query.dtype)

    # add the mask zero out padding tokens.
    if mask is not None:
        logits += (keras.ops.cast(mask, logits.dtype) * -1e9)

    attention_weights = torch.softmax(logits, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights


@keras.utils.register_keras_serializable('GavinCore')
class FourierTransformationLayer(keras.layers.Layer):
    """
    From the paper: https://arxiv.org/pdf/2105.03824.pdf
    Fourier transformations can apparently be used in attention & achieve similar results.
    Applies FFT1D across the first dimension of the embeddings (sequence_length).
    Applies FFT2D across the last two dimensions of the embeddings (sequence_length, d_model).
    Furthermore, applies FFT1D across the last dimension of the embeddings (d_model).

    """

    def __init__(self, name="fourier_transformation", *args, **kwargs):
        super(FourierTransformationLayer, self).__init__(name=name, *args, **kwargs)

    @staticmethod
    def call(inputs: torch.Tensor):
        """
        Args:
            :param inputs: torch.Tensor
                The input tensor to be transformed. Should be of shape (batch_size, sequence_length, d_model)
        :return: torch.Tensor
            The transformed tensor. Should be of shape (batch_size, sequence_length, d_model)
        """
        output = inputs.type(torch.complex64)
        output = torch.fft.fft2(output)
        # output = tf.signal.fft(output)
        # output = tf.signal.fft(output)
        return keras.ops.cast(output, inputs.dtype)


@keras.utils.register_keras_serializable('GavinCore')
# noinspection PyMethodOverriding,PyMethodMayBeStatic
class PositionalEncoding(keras.layers.Layer):
    """Positional Encoding

    Acts as input for the model, attention to where words appear in an input etc...

    Attributes:
        :param position: int
            The position the word appears in
        :param d_model: int
            This is for the attention math, acts as units for other layers in the model too.
    """

    def __init__(self, position: int, d_model: int, **kwargs):
        self.d_model = d_model
        self.position = position
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(position, d_model=d_model)

    def get_angles(self, position: int, i, d_model: int):
        angles = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=torch.arange(0, position)[:, None],
            i=torch.unsqueeze(torch.as_tensor(d_model), 0),
            d_model=d_model)

        # apply sin to even index in the array
        sines = torch.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = torch.cos(angle_rads[:, 1::2])

        pos_encoding = torch.concat((sines, cosines), -1)
        pos_encoding = pos_encoding[None, ...]
        return pos_encoding

    def call(self, inputs):
        # self.pos_encoding = self.pos_encoding.to(inputs.device)
        y = self.pos_encoding[:, :inputs.size()[1], :]
        y = keras.ops.cast(y, inputs.dtype)
        return inputs + y.to(inputs.device)

    def get_config(self):
        cfg = {'d_model': self.d_model,
               'position': self.position}
        return cfg


@keras.utils.register_keras_serializable('GavinCore')
class RotaryPositionalEncoding(keras.layers.Layer):
    """Rotary Positional Encoding
    This kind of positional encoding is used by the GPT-J model, its an alternative to the standard positional encoding
    which is used in the Transformer model. This positional encoding works by adding a sinusoidal signal to the input
    embeddings at the positional positions.
    """

    def __init__(self, name: str = "rotary_positional_encoding", **kwargs):
        super(RotaryPositionalEncoding, self).__init__(name=name, **kwargs)

    @staticmethod
    def align(tensor, axes: List[int], ndim=None):
        """
        https://github.com/bojone/bert4keras/blob/70a7eb9ace18b9f4806b6386e5183f32d024bc37/bert4keras/backend.py#L136
        """
        ndim = ndim or max(axes) + 1
        indices = [None] * ndim
        for i in axes:
            indices[i] = slice(None)
        return tensor[indices]

    def call(self, inputs):
        n = 3
        sinusoidal = self.align(inputs[n], axes=[0, 1, -1], ndim=inputs[0].dim())
        size = sinusoidal.size()[-1]
        cos_pos = torch.repeat_interleave(sinusoidal[..., 1::2], 2, -1, output_size=size)
        sin_pos = torch.repeat_interleave(sinusoidal[..., ::2], 2, -1, output_size=size)
        return inputs * cos_pos + inputs * sin_pos

    def get_config(self):
        return {}


@keras.utils.register_keras_serializable('GavinCore')
# noinspection PyMethodOverriding,PyShadowingNames
class GavinMultiHeadAttention(keras.layers.Layer):
    # noinspection Assert
    def __init__(self, d_model: int, num_heads: int, name: str = "multi_head_attention", **kwargs):
        """Multi Head Attention Layer

        ...
        Attributes:
            :param d_model: int
                Embeddings Size
            :param num_heads: int
                The number of heads the layer should have
            :param name: str
                The name of layer
        """
        super(GavinMultiHeadAttention, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = keras.layers.Dense(units=d_model)
        self.key_dense = keras.layers.Dense(units=d_model)
        self.value_dense = keras.layers.Dense(units=d_model)
        self.saved_attention_image = None

        self.dense = keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size: int):
        inputs = torch.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))  # B, L, H, D
        return inputs.permute(0, 2, 1, 3)  # B, H, L, D

    def call(self, inputs: Dict):
        query, key, value, mask = (inputs['query'], inputs['key'],
                                   inputs['value'], inputs['mask'])
        batch_size = query.size()[0]
        max_len = query.size()[1]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, attention_matrix = scaled_dot_product_attention(query, key, value, mask,
                                                                          name_prefix=self.name)
        self.saved_attention_image = attention_matrix

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)  # B, L, H, depth

        concat_attention = torch.reshape(scaled_attention, (batch_size, max_len, self.d_model))  # B, L, D

        outputs = self.dense(concat_attention)

        return outputs

    def get_config(self):
        cfg = {'d_model': self.d_model,
               'num_heads': self.num_heads}
        return cfg

    def build(self, input_shape):
        super(GavinMultiHeadAttention, self).build(input_shape)


@keras.utils.register_keras_serializable('GavinCore')
class GavinMultiHeadPerformerAttention(GavinMultiHeadAttention):
    """MultiHead attention using the performers' specification,
    significantly improving memory and time complexity allowing for
    higher values of sequence length, whilst maintaining as good or
    some cases better accuracy compared to standard transformer.

    Attributes:
        :param d_model: int
            Embeddings Size
        :param num_heads: int
            The number of heads the layer should have
        :param num_features: int
            Number of features to be used in Gaussian Matrix
        :param name: str
            The name of layer.
    """

    def __init__(self, d_model: int, num_heads: int, num_features: int, name: str = "MultiHeadPerformer", **kwargs):
        self.num_features = num_features
        super().__init__(d_model, num_heads, name, **kwargs)
        self.random_feats = orthogonal_gaussian(self.num_features, self.depth)

    def call(self, inputs: Dict):
        query, key, value = inputs['query'], inputs['key'], inputs['value']

        batch_size = query.size()[0]
        max_len = query.size()[1]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)  # B, H, L, D
        key = self.split_heads(key, batch_size)  # B, H, L, D
        value = self.split_heads(value, batch_size)  # B, H, L, D

        scaled_attention = positive_attention(query=query, key=key, value=value,
                                              random_feats=self.random_feats)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        concat_attention = torch.reshape(scaled_attention, (batch_size, max_len, self.d_model))  # B, L, D

        outputs = self.dense(concat_attention)

        return outputs

    def get_config(self):
        cfg = {'d_model': self.d_model,
               'num_heads': self.num_heads,
               'num_features': self.num_features}
        return cfg

    def build(self, input_shape):
        super(GavinMultiHeadPerformerAttention, self).build(input_shape)


@keras.utils.register_keras_serializable('GavinCore')
class MultiHeadPerformerReluAttention(GavinMultiHeadPerformerAttention):
    def call(self, inputs: Dict):
        query, key, value = inputs['query'], inputs['key'], inputs['value']

        batch_size = query.size()[0]
        max_len = query.size()[1]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)  # B, H, L, D
        key = self.split_heads(key, batch_size)  # B, H, L, D
        value = self.split_heads(value, batch_size)  # B, H, L, D

        scaled_attention = positive_attention(query=query, key=key, value=value,
                                              random_feats=self.random_feats)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        concat_attention = torch.reshape(scaled_attention, (batch_size, max_len, self.d_model))  # B, L, D

        outputs = self.dense(concat_attention)

        return outputs


@keras.utils.register_keras_serializable('GavinCore')
class PaddingMaskLayer(keras.layers.Layer):
    def __init__(self, batch_size: int, max_len: int, name: str = "padding_mask", **kwargs):
        super(PaddingMaskLayer, self).__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.max_len = max_len

    def call(self, inputs: torch.Tensor, **kwargs):
        shape = inputs.size()
        mask = keras.ops.cast((inputs == 0), inputs.dtype)
        # batch_size, 1, 1, sequence_length
        return torch.reshape(mask, (shape[0], 1, 1, shape[1]))

    def get_config(self):
        cfg = {'batch_size': self.batch_size,
                'max_len': self.max_len}
        return cfg


@keras.utils.register_keras_serializable('GavinCore')
class LookAheadMaskLayer(keras.layers.Layer):
    def __init__(self, batch_size: int, max_len: int, name: str = "look_ahead_mask", **kwargs):
        super(LookAheadMaskLayer, self).__init__(name=name, **kwargs)
        self.padding_mask = PaddingMaskLayer(batch_size, max_len)
        self.batch_size = batch_size
        self.max_len = max_len

    def call(self, inputs: torch.Tensor, **kwargs):
        look_ahead_mask = 1 - torch.tril(torch.ones((self.max_len, self.max_len),
                                                    dtype=inputs.dtype, device=inputs.device))
        padding_mask = self.padding_mask(inputs)
        return torch.maximum(look_ahead_mask, padding_mask)

    def build(self, input_shape):
        super(LookAheadMaskLayer, self).build(input_shape)

    def get_config(self):
        cfg = {
            'batch_size': self.batch_size,
            'max_len': self.max_len
        }
        return cfg


# noinspection PyAttributeOutsideInit
class GPUEnabledEmbedding(keras.layers.Embedding):
    """Embedding Layers are forced to run on CPUs which seriously
    hurts training performance this fixes that issue."""

    @tf_utils.shape_type_conversion
    def build(self, _):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name="embeddings",
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
        )
        self.built = True
