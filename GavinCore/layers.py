import typing
from typing import List
from tensorflow.python.keras.utils import tf_utils

from .utils import keras, torch, keras_ops, np, tf
from typing import Dict


def iid_gaussian(m, d):
    """Generate random values that are I.I.D (independent identically distributed)
    :param m: int
        Hidden Dimensions
    :param d: int
        Depth (half the hidden dimensions)"""
    # Keras Ops currently doesn't contain a .randn, so we'll have to convert numpy
    return keras_ops.convert_to_tensor(np.random.randn(m, d))


def orthogonal_gaussian(m: int, d: int):
    """Generate Orthogonal Gaussian distribution's. This is to improve upon MSE (mean squared error)
    inside a performer.
    Args:
        :param m: int
            Hidden Dimensions
        :param d: int
            Depth (half the hidden dimensions)"""

    def orthogonal_square():
        q, _ = keras_ops.qr(iid_gaussian(d, d))
        return keras_ops.transpose(q)  # transpose

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    # matrix = tf.concat(blocks, axis=0)
    matrix = keras_ops.vstack(blocks)
    matrix /= keras_ops.sqrt(keras_ops.convert_to_tensor(keras_ops.add(num_squares, remainder / d)))

    return matrix


def softmax_kernel_transformation(data: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor],
                                  is_query: bool,
                                  projection_matrix: typing.Union[typing.Union[torch.Tensor, tf.Tensor],
                                  typing.Union[torch.Tensor, tf.Tensor]] = None,
                                  numerical_stabilizer=0.000001):
    """Computes random features for the softmax kernel using FAVOR+ mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    :param data: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor]
        input data tensor of the shape [B, L, H, D], where: B - batch dimension,
            L - attention dimensions, H - heads, D - depth
    :param is_query: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor]
        Indicates whether input data is a query oor key tensor
    :param projection_matrix: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor]
        random Gaussian matrix of shape [M, D], where M stands for the
        number of random features and each D x D sub-block has pairwise orthogonal rows
    :param numerical_stabilizer: float
        small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
    projection_matrix = keras_ops.cast(projection_matrix, data.dtype)
    data_normalizer = 1.0 / (
        keras_ops.sqrt(keras_ops.sqrt(keras_ops.shape(data)[-1])))
    data = data_normalizer * data
    ratio = 1.0 / keras_ops.sqrt(keras_ops.shape(projection_matrix)[0])
    # noinspection SpellCheckingInspection
    # This is the kernel transformation
    data_dash = keras_ops.einsum("blhd,md->blhm", data, projection_matrix)
    # Diag Data is masking out the diagonal values. Similar to how the scaled dot product attention works.
    diag_data = keras_ops.square(data)
    dim = len(keras_ops.shape(data)) - 1
    diag_data = keras_ops.sum(diag_data, axis=dim)
    diag_data = diag_data / 2.0
    diag_data = keras_ops.expand_dims(diag_data, axis=dim)
    last_dims_t = len(keras_ops.shape(data_dash)) - 1
    attention_dims_t = len(keras_ops.shape(data_dash)) - 3
    if is_query:
        data_dash = ratio * (
                keras_ops.exp(data_dash - diag_data - keras_ops.max(data_dash, axis=last_dims_t, keepdims=True)[0])
                + numerical_stabilizer)
    else:
        data_dash = ratio * (
                keras_ops.exp(data_dash - diag_data - keras_ops.max(data_dash, axis=last_dims_t, keepdims=True)[0])
                + numerical_stabilizer)

    return data_dash


def relu_kernel_transformation(data: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor],
                               projection_matrix: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor] = None,
                               numerical_stabilizer=0.000001):
    """Computes random features for the ReLU kernel using FAVOR+ mechanism.

    Args:
        :param data: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor]
            input data tensor of the shape [B, L, H, D], where: B - batch dimension,
            L - attention dimensions, H - heads, D - depth
        :param projection_matrix: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor]
            random Gaussian matrix of shape [M, D], where M stands for the
            number of random features and each D x D sub-block has pairwise orthogonal rows
        :param numerical_stabilizer: float
            small positive constant for numerical stability.
    """
    projection_matrix = keras_ops.cast(projection_matrix, data.dtype)
    m = keras_ops.shape(data)[-1]
    m = keras_ops.cast(keras_ops.convert_to_tensor(m), data.dtype)
    data_normalizer = 1.0 / keras_ops.sqrt(m)
    projection_matmul = keras_ops.einsum("blhd,md->blhm", data, projection_matrix)
    return keras_ops.relu(data_normalizer * projection_matmul + numerical_stabilizer)


def attn_hat(query: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor],
             key: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor],
             value: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor],
             phi_fun=None, random_feats: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor] = None):
    """
    Args:
        :param query: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor]
            The Query tensor from the Multi-headed attention mechanism
        :param key: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor]
        The Key tensor from the Multi-headed attention mechanism
        :param value: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor]
            The Value tensor from the Multi-headed attention mechanism
        :param phi_fun: Any function
            A function for "phi" If None, default to Softmax kernel transformations
        :param random_feats: typing.Union[typing.Union[torch.Tensor, tf.Tensor], tf.Tensor]
            The random features for use in phi function in predicting the softmax values
    """
    sequence_length = keras_ops.shape(query)[2]
    # B, H, L, D to B, L, H, D
    query = keras_ops.transpose(query, (0, 2, 1, 3))
    key = keras_ops.transpose(key, (0, 2, 1, 3))
    if phi_fun is not None:
        q_prime = phi_fun(query, random_feats)
        k_prime = phi_fun(key, random_feats)
    else:
        q_prime = softmax_kernel_transformation(query, projection_matrix=random_feats, is_query=True)  # B L H M
        k_prime = softmax_kernel_transformation(key, projection_matrix=random_feats, is_query=False)  # B L H M

    # B H L D, L B H D
    value = keras_ops.transpose(value, (2, 0, 1, 3))

    # B L H M, L B H M
    k_prime = keras_ops.transpose(k_prime, (1, 0, 2, 3))  # L B H M
    q_prime = keras_ops.transpose(q_prime, (1, 0, 2, 3))  # L B H M

    # noinspection SpellCheckingInspection
    av_attention = keras_ops.einsum("lbhm,lbhd->bhmd", k_prime, value)

    # noinspection SpellCheckingInspection
    av_attention = keras_ops.einsum("lbhm,bhmd->lbhd", q_prime, av_attention)
    # noinspection SpellCheckingInspection
    normalizer = keras_ops.einsum("lbhm,l->bhm", k_prime, keras_ops.ones(sequence_length, dtype=k_prime.dtype))
    # noinspection SpellCheckingInspection
    normalizer = keras_ops.einsum("lbhm,bhm->lbh", q_prime, normalizer)
    av_attention = keras_ops.transpose(av_attention, (1, 0, 2, 3))  # B L H D
    normalizer = keras_ops.transpose(normalizer, (1, 0, 2))  # B L H
    normalizer = keras_ops.expand_dims(normalizer, len(keras_ops.shape(normalizer)))  # B L H 1
    return av_attention / normalizer


def positive_attention(query: typing.Union[torch.Tensor, tf.Tensor], key: typing.Union[torch.Tensor, tf.Tensor],
                       value: typing.Union[torch.Tensor, tf.Tensor],
                       random_feats: typing.Union[torch.Tensor, tf.Tensor]):
    """Instead of using ScaledDotProduction, this uses the above Gaussian elements to estimate the answer that
    the full ScaledDotProduction would give.
    Args:
        :param query: typing.Union[torch.Tensor, tf.Tensor]
            The Query tensor from the Multi-headed attention mechanism
        :param key: typing.Union[torch.Tensor, tf.Tensor]
            The Key tensor from the Multi-headed attention mechanism
        :param value:
            The Value tensor from the Multi-headed attention mechanism
        :param random_feats:
            The random features for use in phi function in predicting the softmax values.
        """

    return attn_hat(query, key, value, random_feats=random_feats)


def positive_relu_attention(query: typing.Union[torch.Tensor, tf.Tensor], key: typing.Union[torch.Tensor, tf.Tensor],
                            value: typing.Union[torch.Tensor, tf.Tensor],
                            random_feats: typing.Union[torch.Tensor, tf.Tensor]):
    """Instead of using ScaledDotProduction, this uses the above Gaussian elements to estimate the answer that
    the full ScaledDotProduction would give.
    Args:
        :param query: typing.Union[torch.Tensor, tf.Tensor]
            The Query tensor from the Multi-headed attention mechanism
        :param key: typing.Union[torch.Tensor, tf.Tensor]
            The Key tensor from the Multi-headed attention mechanism
        :param value:
            The Value tensor from the Multi-headed attention mechanism
        :param random_feats:
            The random features for use in phi function in predicting the softmax values.
        """

    return attn_hat(query, key, value, random_feats=random_feats, phi_fun=relu_kernel_transformation)


def scaled_dot_product_attention(query: typing.Union[torch.Tensor, tf.Tensor],
                                 key: typing.Union[torch.Tensor, tf.Tensor],
                                 value: typing.Union[torch.Tensor, tf.Tensor],
                                 mask: typing.Union[torch.Tensor, tf.Tensor], name_prefix: str) \
        -> typing.Tuple[typing.Union[torch.Tensor, tf.Tensor], typing.Union[torch.Tensor, tf.Tensor]]:
    """
    Args:
        :param query: typing.Union[torch.Tensor, tf.Tensor]
            The Query tensor from the Multi-headed attention mechanism
        :param key: typing.Union[torch.Tensor, tf.Tensor]
            The Key tensor from the Multi-headed attention mechanism
        :param value: typing.Union[torch.Tensor, tf.Tensor]
            The Value tensor from the Multi-headed attention mechanism
        :param mask: typing.Union[torch.Tensor, tf.Tensor]
            For masking out previous outputs
        :param name_prefix: str
            The name prefix for the attention mechanism
    :return: The final tensor object
    """

    matmul_qk = keras_ops.matmul(query, keras_ops.transpose(key, (0, 1, 3, 2)))

    depth = keras.ops.cast(keras_ops.shape(key)[-1], query.dtype)
    logits = matmul_qk / keras_ops.sqrt(depth)
    logits = keras.ops.cast(logits, query.dtype)

    # add the mask zero out padding tokens.
    if mask is not None:
        logits += (keras.ops.cast(mask, logits.dtype) * -1e9)

    attention_weights = keras_ops.softmax(logits, axis=-1)
    return keras_ops.matmul(attention_weights, value), attention_weights


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
    def call(inputs: typing.Union[torch.Tensor, tf.Tensor]):
        """
        Args:
            :param inputs: typing.Union[torch.Tensor, tf.Tensor]
                The input tensor to be transformed. Should be of shape (batch_size, sequence_length, d_model)
        :return: typing.Union[torch.Tensor, tf.Tensor]
            The transformed tensor. Should be of shape (batch_size, sequence_length, d_model)
        """
        # output = keras_ops.cast(inputs, 'complex64')
        output = keras_ops.fft2(inputs)
        # output = tf.signal.fft(output)
        # output = tf.signal.fft(output)
        return keras_ops.cast(output, inputs.dtype)


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
        angles = 1 / keras_ops.power(10000, (2 * (i // 2)) / d_model)
        return keras_ops.multiply(position, angles)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=keras_ops.arange(0, position)[:, None],
            i=keras_ops.expand_dims(keras_ops.convert_to_tensor(d_model), 0),
            d_model=d_model)

        # apply sin to even index in the array
        sines = keras_ops.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = keras_ops.cos(angle_rads[:, 1::2])

        pos_encoding = keras_ops.concatenate((sines, cosines), -1)
        pos_encoding = pos_encoding[None, ...]
        return pos_encoding

    def call(self, inputs):
        # self.pos_encoding = self.pos_encoding.to(inputs.device)
        y = self.pos_encoding[:, :keras_ops.shape(inputs)[1], :]
        y = keras.ops.cast(y, inputs.dtype)
        return keras_ops.add(inputs, y)

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
        sinusoidal = self.align(inputs[n], axes=[0, 1, -1], ndim=len(keras_ops.shape(inputs[0])))
        size = keras_ops.shape(sinusoidal)[-1]
        cos_pos = keras_ops.repeat(sinusoidal[..., 1::2], 2, -1)
        sin_pos = keras_ops.repeat(sinusoidal[..., ::2], 2, -1)
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
        inputs = keras_ops.reshape(inputs, (batch_size, -1, self.num_heads, self.depth))  # B, L, H, D
        return keras_ops.transpose(inputs, (0, 2, 1, 3))  # B, H, L, D

    def call(self, inputs: Dict):
        query, key, value, mask = (inputs['query'], inputs['key'],
                                   inputs['value'], inputs['mask'])
        batch_size = keras_ops.shape(query)[0]
        max_len = keras_ops.shape(query)[1]

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

        scaled_attention = keras_ops.transpose(scaled_attention, (0, 2, 1, 3))  # B, L, H, depth

        concat_attention = keras_ops.reshape(scaled_attention, (batch_size, max_len, self.d_model))  # B, L, D

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

        batch_size = keras_ops.shape(query)[0]
        max_len = keras_ops.shape(query)[1]

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

        scaled_attention = keras_ops.transpose(scaled_attention, (0, 2, 1, 3))

        concat_attention = keras_ops.reshape(scaled_attention, (batch_size, max_len, self.d_model))  # B, L, D

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

        batch_size = keras_ops.shape(query)[0]
        max_len = keras_ops.shape(query)[1]

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

        scaled_attention = keras_ops.transpose(scaled_attention, (0, 2, 1, 3))

        concat_attention = keras_ops.reshape(scaled_attention, (batch_size, max_len, self.d_model))  # B, L, D

        outputs = self.dense(concat_attention)

        return outputs


@keras.utils.register_keras_serializable('GavinCore')
class PaddingMaskLayer(keras.layers.Layer):
    def __init__(self, batch_size: int, max_len: int, name: str = "padding_mask", **kwargs):
        super(PaddingMaskLayer, self).__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.max_len = max_len

    def call(self, inputs: typing.Union[torch.Tensor, tf.Tensor], **kwargs):
        shape = keras_ops.shape(inputs)
        mask = keras.ops.cast((inputs == 0), inputs.dtype)
        # batch_size, 1, 1, sequence_length
        return keras_ops.reshape(mask, (shape[0], 1, 1, shape[1]))

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

    def call(self, inputs: typing.Union[torch.Tensor, tf.Tensor], **kwargs):
        look_ahead_mask = 1 - keras_ops.tril(keras_ops.ones((self.max_len, self.max_len), dtype=inputs.dtype))
        padding_mask = self.padding_mask(inputs)
        return keras_ops.maximum(look_ahead_mask, padding_mask)

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
