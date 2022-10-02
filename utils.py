from tensorflow.python.ops.rnn import rnn_cell_impl, _should_cache, nest, vs, tensor_shape, _is_keras_rnn_cell, ops, _concat, array_ops, math_ops, _infer_state_dtype, dtypes, _rnn_step
from collections import deque
import random
import numpy as np
def my_static_rnn(cell,
               inputs,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
  """Creates a recurrent neural network specified by RNNCell `cell`.
  The simplest form of RNN network generated is:
  ```python
    state = cell.zero_state(...)
    outputs = []
    for input_ in inputs:
      output, state = cell(input_, state)
      outputs.append(output)
    return (outputs, state)
  ```
  However, a few other options are available:
  An initial state can be provided.
  If the sequence_length vector is provided, dynamic calculation is performed.
  This method of calculation does not compute the RNN steps past the maximum
  sequence length of the minibatch (thus saving computational time),
  and properly propagates the state at an example's sequence length
  to the final state output.
  The dynamic calculation performed is, at time `t` for batch row `b`,
  ```python
    (output, state)(b, t) =
      (t >= sequence_length(b))
        ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
        : cell(input(b, t), state(b, t - 1))
  ```
  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a `Tensor` of shape `[batch_size,
      input_size]`, or a nested tuple of such elements.
    initial_state: (optional) An initial state for the RNN. If `cell.state_size`
      is an integer, this must be a `Tensor` of appropriate type and shape
      `[batch_size, cell.state_size]`. If `cell.state_size` is a tuple, this
      should be a tuple of tensors having shapes `[batch_size, s] for s in
      cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    sequence_length: Specifies the length of each sequence in inputs. An int32
      or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
    scope: VariableScope for the created subgraph; defaults to "rnn".
  Returns:
    A pair (outputs, state) where:
    - outputs is a length T list of outputs (one for each input), or a nested
      tuple of such elements.
    - state is the final state
  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If `inputs` is `None` or an empty list, or if the input depth
      (column size) cannot be inferred from inputs via shape inference.
  """
  rnn_cell_impl.assert_like_rnncell("cell", cell)
  if not nest.is_sequence(inputs):
    raise TypeError("inputs must be a sequence")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []
  cell_states = []

  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with vs.variable_scope(scope or "rnn") as varscope:
    if _should_cache():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    # Obtain the first sequence of the input
    first_input = inputs
    while nest.is_sequence(first_input):
      first_input = first_input[0]

    # Temporarily avoid EmbeddingWrapper and seq2seq badness
    # TODO(lukaszkaiser): remove EmbeddingWrapper
    if first_input.get_shape().rank != 1:

      input_shape = first_input.get_shape().with_rank_at_least(2)
      fixed_batch_size = input_shape.dims[0]

      flat_inputs = nest.flatten(inputs)
      for flat_input in flat_inputs:
        input_shape = flat_input.get_shape().with_rank_at_least(2)
        batch_size, input_size = tensor_shape.dimension_at_index(
            input_shape, 0), input_shape[1:]
        fixed_batch_size.merge_with(batch_size)
        for i, size in enumerate(input_size.dims):
          if tensor_shape.dimension_value(size) is None:
            raise ValueError(
                "Input size (dimension %d of inputs) must be accessible via "
                "shape inference, but saw value None." % i)
    else:
      fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

    if tensor_shape.dimension_value(fixed_batch_size):
      batch_size = tensor_shape.dimension_value(fixed_batch_size)
    else:
      batch_size = array_ops.shape(first_input)[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, "
                         "dtype must be specified")
      if getattr(cell, "get_initial_state", None) is not None:
        state = cell.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=dtype)
      else:
        state = cell.zero_state(batch_size, dtype)

    if sequence_length is not None:  # Prepare variables
      sequence_length = ops.convert_to_tensor(
          sequence_length, name="sequence_length")
      if sequence_length.get_shape().rank not in (None, 1):
        raise ValueError(
            "sequence_length must be a vector of length batch_size")

      def _create_zero_output(output_size):
        # convert int to TensorShape if necessary
        size = _concat(batch_size, output_size)
        output = array_ops.zeros(
            array_ops.stack(size), _infer_state_dtype(dtype, state))
        shape = _concat(
            tensor_shape.dimension_value(fixed_batch_size),
            output_size,
            static=True)
        output.set_shape(tensor_shape.TensorShape(shape))
        return output

      output_size = cell.output_size
      flat_output_size = nest.flatten(output_size)
      flat_zero_output = tuple(
          _create_zero_output(size) for size in flat_output_size)
      zero_output = nest.pack_sequence_as(
          structure=output_size, flat_sequence=flat_zero_output)

      sequence_length = math_ops.cast(sequence_length, dtypes.int32)
      min_sequence_length = math_ops.reduce_min(sequence_length)
      max_sequence_length = math_ops.reduce_max(sequence_length)

    # Keras RNN cells only accept state as list, even if it's a single tensor.
    is_keras_rnn_cell = _is_keras_rnn_cell(cell)
    if is_keras_rnn_cell and not nest.is_sequence(state):
      state = [state]
    for time, input_ in enumerate(inputs):
      if time > 0:
        varscope.reuse_variables()
      # pylint: disable=cell-var-from-loop
      call_cell = lambda: cell(input_, state)
      # pylint: enable=cell-var-from-loop
      if sequence_length is not None:
        (output, state) = _rnn_step(
            time=time,
            sequence_length=sequence_length,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            zero_output=zero_output,
            state=state,
            call_cell=call_cell,
            state_size=cell.state_size)
      else:
        (output, state) = call_cell()
      outputs.append(output)
      cell_states.append(state[0])
    # Keras RNN cells only return state as list, even if it's a single tensor.
    if is_keras_rnn_cell and len(state) == 1:
      state = state[0]

    return (outputs, state, cell_states)


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed = 123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []
        random.seed(random_seed)

    def add(self, s, a, r, s2):
        experience = (s, a, r, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            
    def add_batch(self, sars2_list):
#         assert self.count + len(sars2_list) <= self.buffer_size, "ER Buffer Overflowed"
        if self.count + len(sars2_list) < self.buffer_size: 
            self.buffer = self.buffer + sars2_list
            self.count += len(sars2_list)
        else:
            del self.buffer[:len(self.buffer)//2]
            self.buffer = self.buffer + sars2_list
            self.count = len(self.buffer)
        

    def size(self):
        return self.count

    def sample_batch(self, batch_size):

        batch = []

        if self.count < batch_size:
            ran_num = np.arange(self.count)
            batch = list(self.buffer)
        else:
            ran_num = np.random.choice(self.count, batch_size, replace = False)
            batch = [self.buffer[i] for i in ran_num]

        s_batch = []
        a_batch = []
        r_batch = []
        s2_batch = []
        
        for b in batch:
            s_batch += [b[0]]
            a_batch += [b[1]]
            r_batch += [b[2]]
            s2_batch += [b[3]]
        
        s_batch, a_batch, r_batch, s2_batch = np.array(s_batch), np.array(a_batch), np.array(r_batch), np.array(s2_batch)
        
#         s_batch = np.array([_[0] for _ in batch])
#         a_batch = np.array([_[1] for _ in batch])
#         r_batch = np.array([_[2] for _ in batch])
#         s2_batch = np.array([_[3] for _ in batch])
        # index = np.array([_[0] for _ in batch])

        return s_batch, a_batch, r_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0