import h5py
import numpy as np
import heartloc_model
from tensorflow.python.keras import backend as K


def convert_stage1_h5():
    in_h5 = '/nfs/masi/xuk9/src/DeepCAC/data/step1_heartloc/model_weights/step1_heartloc_model_weights.hdf5.bak'
    out_h5 = '/nfs/masi/xuk9/src/DeepCAC/data/step1_heartloc/model_weights/step1_heartloc_model_weights.updated.hdf5'

    print("Load %s" % in_h5)
    print("Output %s" % out_h5)

    in_db = h5py.File(in_h5, 'r')
    out_db = h5py.File(out_h5, 'w')

    in_g = in_db['model_1']
    # layer_names = in_g.keys()
    # out_db.attrs['layer_names'] = layer_names

    # Retrieve the layer information directly from the model
    crop_size = 112
    model_down_steps = 4
    extended = False
    mgpu = 1
    input_shape = (crop_size, crop_size, crop_size, 1)
    model = heartloc_model.get_unet_3d(down_steps=model_down_steps,
                                       input_shape=input_shape,
                                       mgpu=mgpu,
                                       ext=extended)

    filtered_layers = []
    layer_names = []
    for layer in model.layers:
        weights = _legacy_weights(layer)
        if weights:
            filtered_layers.append(layer)
            layer_names.append(layer.name)

    out_db.attrs['layer_names'] = layer_names
    out_db.attrs['backend'] = 'tensorflow'
    out_db.attrs['keras_version'] = '2.0.8'

    # for layer_name, layer_group in in_g.items():
    for k, layer_name in enumerate(layer_names):
        layer_g = out_db.create_group(layer_name)
        # layer_weight_names = in_g[layer_name].keys()
        # layer_group = in_g[layer_name]
        weights = filtered_layers[k].weights
        weight_names = [weight.name for weight in weights]
        weight_names = [name.replace(str(layer_name) + '/', '') for name in weight_names]

        layer_g.attrs['weight_names'] = weight_names
        for weight_name, weight_dataset in in_g[layer_name].items():
            layer_g.create_dataset(weight_name, data=np.asarray(weight_dataset))

    in_db.close()
    out_db.close()


def _legacy_weights(model):
  """DO NOT USE.

  For legacy reason, the model.weights was in the order of
  [self.trainable_weights + self.non_trainable_weights], and this order was
  used for preserving the weights in h5 format. The new order of model.weights
  are the same as model.get_weights() which is more intuitive for user. To
  keep supporting the existing saved h5 file, this method should be used to
  save/load weights. In future version, we will delete this method and
  introduce a breaking change for h5 and stay with the new order for weights.

  Args:
    model: a model or layer instance.

  Returns:
    A list of variables with the order of trainable_weights, followed by
      non_trainable_weights.
  """
  return model.trainable_weights + model.non_trainable_weights


def load_weights_from_hdf5_group(f, layers):
  """Implements topological (order-based) weight loading.

  Arguments:
      f: A pointer to a HDF5 group.
      layers: a list of target layers.

  Raises:
      ValueError: in case of mismatch between provided layers
          and weights file.
  """
  if 'keras_version' in f.attrs:
    original_keras_version = f.attrs['keras_version'].decode('utf8')
  else:
    original_keras_version = '1'
  if 'backend' in f.attrs:
    original_backend = f.attrs['backend'].decode('utf8')
  else:
    original_backend = None

  filtered_layers = []
  for layer in layers:
    weights = _legacy_weights(layer)
    if weights:
      filtered_layers.append(layer)

  layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
  filtered_layer_names = []
  for name in layer_names:
    g = f[name]
    weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
    if weight_names:
      filtered_layer_names.append(name)
  layer_names = filtered_layer_names
  if len(layer_names) != 19:
    raise ValueError('You are trying to load a weight file '
                     'containing ' + str(len(layer_names)) +
                     ' layers into a model with ' + str(19) +
                     ' layers.')

  # We batch weight value assignments in a single backend call
  # which provides a speedup in TensorFlow.
  weight_value_tuples = []
  for k, name in enumerate(layer_names):
    g = f[name]
    weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
    weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
    layer = filtered_layers[k]
    symbolic_weights = _legacy_weights(layer)
    weight_values = preprocess_weights_for_loading(
        layer, weight_values, original_keras_version, original_backend)
    if len(weight_values) != len(symbolic_weights):
      raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                       '" in the current model) was found to '
                       'correspond to layer ' + name + ' in the save file. '
                       'However the new layer ' + layer.name + ' expects ' +
                       str(len(symbolic_weights)) +
                       ' weights, but the saved weights have ' +
                       str(len(weight_values)) + ' elements.')
    weight_value_tuples += zip(symbolic_weights, weight_values)
  K.batch_set_value(weight_value_tuples)


def load_attributes_from_hdf5_group(group, name):
  """Loads attributes of the specified name from the HDF5 group.

  This method deals with an inherent problem
  of HDF5 file which is not able to store
  data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

  Arguments:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to load.

  Returns:
      data: Attributes data.
  """
  if name in group.attrs:
    data = [n.decode('utf8') for n in group.attrs[name]]
  else:
    data = []
    chunk_id = 0
    while '%s%d' % (name, chunk_id) in group.attrs:
      data.extend(
          [n.decode('utf8') for n in group.attrs['%s%d' % (name, chunk_id)]])
      chunk_id += 1
  return data


def check_h5_structure():
    in_h5 = '/nfs/masi/xuk9/src/DeepCAC/data/step1_heartloc/model_weights/step1_heartloc_model_weights.updated.hdf5'

    crop_size = 112
    model_down_steps = 4
    extended = False
    mgpu = 1
    input_shape = (crop_size, crop_size, crop_size, 1)
    model = heartloc_model.get_unet_3d(down_steps=model_down_steps,
                                       input_shape=input_shape,
                                       mgpu=mgpu,
                                       ext=extended)

    db = h5py.File(in_h5, 'r')

    load_weights_from_hdf5_group(db, model.layers)

    db.close()



def preprocess_weights_for_loading(layer,
                                   weights,
                                   original_keras_version=None,
                                   original_backend=None):
  """Preprocess layer weights between different Keras formats.

  Converts layers weights from Keras 1 format to Keras 2 and also weights of
  CuDNN layers in Keras 2.

  Arguments:
      layer: Layer instance.
      weights: List of weights values (Numpy arrays).
      original_keras_version: Keras version for the weights, as a string.
      original_backend: Keras backend the weights were trained with,
          as a string.

  Returns:
      A list of weights values (Numpy arrays).
  """
  def convert_nested_bidirectional(weights):
    """Converts layers nested in `Bidirectional` wrapper.

    This function uses `preprocess_weights_for_loading()` for converting
    layers.

    Arguments:
        weights: List of weights values (Numpy arrays).

    Returns:
        A list of weights values (Numpy arrays).
    """
    num_weights_per_layer = len(weights) // 2
    forward_weights = preprocess_weights_for_loading(
        layer.forward_layer, weights[:num_weights_per_layer],
        original_keras_version, original_backend)
    backward_weights = preprocess_weights_for_loading(
        layer.backward_layer, weights[num_weights_per_layer:],
        original_keras_version, original_backend)
    return forward_weights + backward_weights

  def convert_nested_time_distributed(weights):
    """Converts layers nested in `TimeDistributed` wrapper.

    This function uses `preprocess_weights_for_loading()` for converting nested
    layers.

    Arguments:
        weights: List of weights values (Numpy arrays).

    Returns:
        A list of weights values (Numpy arrays).
    """
    return preprocess_weights_for_loading(
        layer.layer, weights, original_keras_version, original_backend)

  def convert_nested_model(weights):
    """Converts layers nested in `Model` or `Sequential`.

    This function uses `preprocess_weights_for_loading()` for converting nested
    layers.

    Arguments:
        weights: List of weights values (Numpy arrays).

    Returns:
        A list of weights values (Numpy arrays).
    """
    new_weights = []
    # trainable weights
    for sublayer in layer.layers:
      num_weights = len(sublayer.trainable_weights)
      if num_weights > 0:
        new_weights.extend(preprocess_weights_for_loading(
            layer=sublayer,
            weights=weights[:num_weights],
            original_keras_version=original_keras_version,
            original_backend=original_backend))
        weights = weights[num_weights:]

    # non-trainable weights
    for sublayer in layer.layers:
      num_weights = len([l for l in sublayer.weights
                         if l not in sublayer.trainable_weights])
      if num_weights > 0:
        new_weights.extend(preprocess_weights_for_loading(
            layer=sublayer,
            weights=weights[:num_weights],
            original_keras_version=original_keras_version,
            original_backend=original_backend))
        weights = weights[num_weights:]
    return new_weights

  # Convert layers nested in Bidirectional/Model/Sequential.
  # Both transformation should be ran for both Keras 1->2 conversion
  # and for conversion of CuDNN layers.
  if layer.__class__.__name__ == 'Bidirectional':
    weights = convert_nested_bidirectional(weights)
  if layer.__class__.__name__ == 'TimeDistributed':
    weights = convert_nested_time_distributed(weights)
  elif layer.__class__.__name__ in ['Model', 'Sequential']:
    weights = convert_nested_model(weights)

  if original_keras_version == '1':
    if layer.__class__.__name__ == 'TimeDistributed':
      weights = preprocess_weights_for_loading(
          layer.layer, weights, original_keras_version, original_backend)

    if layer.__class__.__name__ == 'Conv1D':
      shape = weights[0].shape
      # Handle Keras 1.1 format
      if shape[:2] != (layer.kernel_size[0], 1) or shape[3] != layer.filters:
        # Legacy shape:
        # (filters, input_dim, filter_length, 1)
        assert shape[0] == layer.filters and shape[2:] == (layer.kernel_size[0],
                                                           1)
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
      weights[0] = weights[0][:, 0, :, :]

    if layer.__class__.__name__ == 'Conv2D':
      if layer.data_format == 'channels_first':
        # old: (filters, stack_size, kernel_rows, kernel_cols)
        # new: (kernel_rows, kernel_cols, stack_size, filters)
        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

    if layer.__class__.__name__ == 'Conv2DTranspose':
      if layer.data_format == 'channels_last':
        # old: (kernel_rows, kernel_cols, stack_size, filters)
        # new: (kernel_rows, kernel_cols, filters, stack_size)
        weights[0] = np.transpose(weights[0], (0, 1, 3, 2))
      if layer.data_format == 'channels_first':
        # old: (filters, stack_size, kernel_rows, kernel_cols)
        # new: (kernel_rows, kernel_cols, filters, stack_size)
        weights[0] = np.transpose(weights[0], (2, 3, 0, 1))

    if layer.__class__.__name__ == 'Conv3D':
      if layer.data_format == 'channels_first':
        # old: (filters, stack_size, ...)
        # new: (..., stack_size, filters)
        weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))

    if layer.__class__.__name__ == 'GRU':
      if len(weights) == 9:
        kernel = np.concatenate([weights[0], weights[3], weights[6]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[4], weights[7]], axis=-1)
        bias = np.concatenate([weights[2], weights[5], weights[8]], axis=-1)
        weights = [kernel, recurrent_kernel, bias]

    if layer.__class__.__name__ == 'LSTM':
      if len(weights) == 12:
        # old: i, c, f, o
        # new: i, f, c, o
        kernel = np.concatenate(
            [weights[0], weights[6], weights[3], weights[9]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[7], weights[4], weights[10]], axis=-1)
        bias = np.concatenate(
            [weights[2], weights[8], weights[5], weights[11]], axis=-1)
        weights = [kernel, recurrent_kernel, bias]

    if layer.__class__.__name__ == 'ConvLSTM2D':
      if len(weights) == 12:
        kernel = np.concatenate(
            [weights[0], weights[6], weights[3], weights[9]], axis=-1)
        recurrent_kernel = np.concatenate(
            [weights[1], weights[7], weights[4], weights[10]], axis=-1)
        bias = np.concatenate(
            [weights[2], weights[8], weights[5], weights[11]], axis=-1)
        if layer.data_format == 'channels_first':
          # old: (filters, stack_size, kernel_rows, kernel_cols)
          # new: (kernel_rows, kernel_cols, stack_size, filters)
          kernel = np.transpose(kernel, (2, 3, 1, 0))
          recurrent_kernel = np.transpose(recurrent_kernel, (2, 3, 1, 0))
        weights = [kernel, recurrent_kernel, bias]

  conv_layers = ['Conv1D', 'Conv2D', 'Conv3D', 'Conv2DTranspose', 'ConvLSTM2D']
  if layer.__class__.__name__ in conv_layers:
    if K.int_shape(layer.weights[0]) != weights[0].shape:
      weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
      if layer.__class__.__name__ == 'ConvLSTM2D':
        weights[1] = np.transpose(weights[1], (3, 2, 0, 1))

  # convert CuDNN layers
  return _convert_rnn_weights(layer, weights)


def _convert_rnn_weights(layer, weights):
  """Converts weights for RNN layers between native and CuDNN format.

  Input kernels for each gate are transposed and converted between Fortran
  and C layout, recurrent kernels are transposed. For LSTM biases are summed/
  split in half, for GRU biases are reshaped.

  Weights can be converted in both directions between `LSTM` and`CuDNNSLTM`
  and between `CuDNNGRU` and `GRU(reset_after=True)`. Default `GRU` is not
  compatible with `CuDNNGRU`.

  For missing biases in `LSTM`/`GRU` (`use_bias=False`) no conversion is made.

  Arguments:
      layer: Target layer instance.
      weights: List of source weights values (input kernels, recurrent
          kernels, [biases]) (Numpy arrays).

  Returns:
      A list of converted weights values (Numpy arrays).

  Raises:
      ValueError: for incompatible GRU layer/weights or incompatible biases
  """

  def transform_kernels(kernels, func, n_gates):
    """Transforms kernel for each gate separately using given function.

    Arguments:
        kernels: Stacked array of kernels for individual gates.
        func: Function applied to kernel of each gate.
        n_gates: Number of gates (4 for LSTM, 3 for GRU).

    Returns:
        Stacked array of transformed kernels.
    """
    return np.hstack([func(k) for k in np.hsplit(kernels, n_gates)])

  def transpose_input(from_cudnn):
    """Makes a function that transforms input kernels from/to CuDNN format.

    It keeps the shape, but changes between the layout (Fortran/C). Eg.:

    ```
    Keras                 CuDNN
    [[0, 1, 2],  <--->  [[0, 2, 4],
     [3, 4, 5]]          [1, 3, 5]]
    ```

    It can be passed to `transform_kernels()`.

    Arguments:
        from_cudnn: `True` if source weights are in CuDNN format, `False`
            if they're in plain Keras format.

    Returns:
        Function that converts input kernel to the other format.
    """
    order = 'F' if from_cudnn else 'C'

    def transform(kernel):
      return kernel.T.reshape(kernel.shape, order=order)

    return transform

  target_class = layer.__class__.__name__

  # convert the weights between CuDNNLSTM and LSTM
  if target_class in ['LSTM', 'CuDNNLSTM'] and len(weights) == 3:
    # determine if we're loading a CuDNNLSTM layer
    # from the number of bias weights:
    # CuDNNLSTM has (units * 8) weights; while LSTM has (units * 4)
    # if there's no bias weight in the file, skip this conversion
    units = weights[1].shape[0]
    bias_shape = weights[2].shape
    n_gates = 4

    if bias_shape == (2 * units * n_gates,):
      source = 'CuDNNLSTM'
    elif bias_shape == (units * n_gates,):
      source = 'LSTM'
    else:
      raise ValueError('Invalid bias shape: ' + str(bias_shape))

    def convert_lstm_weights(weights, from_cudnn=True):
      """Converts the weights between CuDNNLSTM and LSTM.

      Arguments:
        weights: Original weights.
        from_cudnn: Indicates whether original weights are from CuDNN layer.

      Returns:
        Updated weights compatible with LSTM.
      """

      # Transpose (and reshape) input and recurrent kernels
      kernels = transform_kernels(weights[0], transpose_input(from_cudnn),
                                  n_gates)
      recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
      if from_cudnn:
        # merge input and recurrent biases into a single set
        biases = np.sum(np.split(weights[2], 2, axis=0), axis=0)
      else:
        # Split single set of biases evenly to two sets. The way of
        # splitting doesn't matter as long as the two sets sum is kept.
        biases = np.tile(0.5 * weights[2], 2)
      return [kernels, recurrent_kernels, biases]

    if source != target_class:
      weights = convert_lstm_weights(weights, from_cudnn=source == 'CuDNNLSTM')

  # convert the weights between CuDNNGRU and GRU(reset_after=True)
  if target_class in ['GRU', 'CuDNNGRU'] and len(weights) == 3:
    # We can determine the source of the weights from the shape of the bias.
    # If there is no bias we skip the conversion since
    # CuDNNGRU always has biases.

    units = weights[1].shape[0]
    bias_shape = weights[2].shape
    n_gates = 3

    def convert_gru_weights(weights, from_cudnn=True):
      """Converts the weights between CuDNNGRU and GRU.

      Arguments:
        weights: Original weights.
        from_cudnn: Indicates whether original weights are from CuDNN layer.

      Returns:
        Updated weights compatible with GRU.
      """

      kernels = transform_kernels(weights[0], transpose_input(from_cudnn),
                                  n_gates)
      recurrent_kernels = transform_kernels(weights[1], lambda k: k.T, n_gates)
      biases = np.array(weights[2]).reshape((2, -1) if from_cudnn else -1)
      return [kernels, recurrent_kernels, biases]

    if bias_shape == (2 * units * n_gates,):
      source = 'CuDNNGRU'
    elif bias_shape == (2, units * n_gates):
      source = 'GRU(reset_after=True)'
    elif bias_shape == (units * n_gates,):
      source = 'GRU(reset_after=False)'
    else:
      raise ValueError('Invalid bias shape: ' + str(bias_shape))

    if target_class == 'CuDNNGRU':
      target = 'CuDNNGRU'
    elif layer.reset_after:
      target = 'GRU(reset_after=True)'
    else:
      target = 'GRU(reset_after=False)'

    # only convert between different types
    if source != target:
      types = (source, target)
      if 'GRU(reset_after=False)' in types:
        raise ValueError('%s is not compatible with %s' % types)
      if source == 'CuDNNGRU':
        weights = convert_gru_weights(weights, from_cudnn=True)
      elif source == 'GRU(reset_after=True)':
        weights = convert_gru_weights(weights, from_cudnn=False)

  return weights


if __name__ == '__main__':
    # convert_stage1_h5()
    check_h5_structure()
