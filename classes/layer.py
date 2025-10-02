from abc import ABC, abstractmethod

import numpy as np

class Layer(ABC):

    def __init__(
            self,
            activation_fn=None,
            activation_fn_dx=None
        ):
        self._input = None
        self._output = None

        # Set activation functions
        self._activation_fn: function = activation_fn
        self._activation_fn_dx: function = activation_fn_dx

        # Define Z values
        self._z_values: np.ndarray = None

        # Adam terms
        self._mw = None
        self._mb = None
        self._vw = None
        self._vb = None
        self._m_hat_w = None
        self._v_hat_w = None
        self._m_hat_b = None
        self._v_hat_b = None

    # GETTER METHODS
    def get_input(self) -> np.ndarray:
        return self._input
    
    def get_output(self) -> np.ndarray:
        return self._output

    @abstractmethod
    def get_input_size(self):
        pass

    @abstractmethod
    def get_output_size(self):
        pass

    def apply_activation_fn_dx(self):
        return self._activation_fn_dx(self._z_values.copy())

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def update_layer(
        self, 
        weights_grad: np.ndarray, 
        bias_grad: np.ndarray, 
        lr: float, 
        epoch_num: int, 
        clip_value=1.0, 
        beta_1=None, 
        beta_2=None,
        l2_lambda=0
    ):
        pass

    # Made with assistance from https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    # Made with assistance from Kingma & Lei Ba (2015) - https://arxiv.org/pdf/1412.6980
    def update_velocity(self, weights_grad, bias_grad, beta_1, beta_2, t):
        # Get all the terms for Adam - mw, mb, vw, vb. mw and mb are the "first moment estimates" and vw and vb are the "second raw moment estimates"
        if self._mw is None and self._mb is None and self._vw is None and self._vb is None:
            self._mw = np.zeros_like(weights_grad, dtype=np.float64)
            self._mb = np.zeros_like(bias_grad, dtype=np.float64)
            self._vw = np.zeros_like(weights_grad, dtype=np.float64)
            self._vb = np.zeros_like(bias_grad) if isinstance(bias_grad, np.ndarray) else 0
        
        # Set velocity terms
        self._mw = beta_1 * self._mw + (1 - beta_1) * weights_grad
        self._mb = beta_1 * self._mb + (1 - beta_1) * bias_grad
        self._vw = beta_2 * self._vw + (1 - beta_2) * weights_grad * weights_grad
        self._vb = beta_2 * self._vb + (1 - beta_2) * bias_grad * bias_grad

        # Set m_hat and v_hat
        self._m_hat_w = self._mw * (1 / (1 - np.power(beta_1, t)))
        self._m_hat_b = self._mb * (1 / (1 - np.power(beta_1, t)))
        self._v_hat_w = self._vw * (1 / (1 - np.power(beta_2, t)))
        self._v_hat_b = self._vb * (1 / (1 - np.power(beta_2, t)))
    
    @abstractmethod
    def contains_nan(self):
        pass



class ConvolutionalLayer(Layer):
    # CONSTRUCTOR
    def __init__(
            self,
            kernel_size:int=3,
            stride:int=1,
            padding:int=0,
            activation_fn=None,
            activation_fn_dx=None,
            input_dims:tuple=(28, 28),
        ):
        super().__init__(activation_fn, activation_fn_dx)

        # Initialize parameters
        self._K: int = kernel_size # the length and width of the kernel filters
        self._S: int = stride # the stride applied during convolution
        self._P: int = padding # the amount of padding to add to the outside of the image
        self._input_dims: tuple = input_dims # the dimensions of the input supplied to the layer

        # Initialize kernels
        self._kernel: np.ndarray = self._he_initialize_kernel(kernel_size)
        self._bias: float = self._initialize_biases()


    # GETTER METHODS
    def get_kernel(self) -> np.ndarray:
        return self._kernel
    
    def get_rotated_kernel(self) -> np.ndarray:
        flattened_kernel = self._kernel.flatten()
        flattened_kernel = np.flip(flattened_kernel)
        return flattened_kernel.reshape((self._K, self._K))
    
    def get_output_shape(self) -> tuple:
        return self._output.shape
    
    def get_kernel_shape(self) -> np.ndarray:
        return self._kernel.shape
    
    def get_input_size(self):
        return self._input_dims
    
    def get_stride(self):
        return self._S
    
    def contains_nan(self):
        return np.any(np.isnan(self._kernel))
    

    # SETTER METHODS
    def set_kernels(self, kernels: np.ndarray):
        if not kernels.shape == self._kernel.shape:
            raise ValueError("Error: kernel shapes do not match")
        self._kernel = kernels

    def set_biases(self, biases: np.ndarray):
        if not self._bias.shape == biases.shape:
            raise ValueError("Error: bias values arrays shapes do not match")
        self._bias = biases 
    
    def set_input_dims(self, input_dims: tuple):
        self._input_dims = input_dims


    # FUNCTIONAL METHODS
    @abstractmethod
    def forward(self, x: np.ndarray):
        pass

    def convolve(
            x:np.ndarray, 
            kernel:np.ndarray, 
            bias:float=None, 
            p:int=0, 
            s:int=1
        ) -> np.ndarray:
        # Apply padding
        h, w = x.shape # number of rows, number of columns, number of channels
        K = len(kernel)
        padded_x = np.zeros((w + 2 * p, h + 2 * p))
        if p != 0:
            padded_x[p:-p, p:-p] = x[:, :]
        else:
            padded_x = x

        # Initialize input and output variables
        output = np.zeros((int((h - K + 2 * p) / s) + 1, int((w - K + 2 * p) / s) + 1))
        out_h, out_w= output.shape
        
        # Iterate through the rows and columns of the output in layer c
        for row in range(out_h):
            for col in range(out_w):
                # Get block to convolve
                start_row = row * s
                end_row = start_row + K
                start_col = col * s
                end_col = start_col + K
                block = padded_x[start_row:end_row, start_col:end_col] # pixel rows, pixel columns, colour values for pixel

                # Convolve input and add bias
                curr_block_layer_vals = block[:, :]
                val = np.sum(np.multiply(kernel, curr_block_layer_vals))
                val += bias if bias is not None else 0
                output[row, col] = val

        # Return output
        return output
    
    def dilate(kernel: np.ndarray, dilation=1):
        # Get shape of kernel
        h, w = kernel.shape

        # Get size of dilated kernel
        kernel_out = np.zeros(((h * (dilation + 1)) - dilation, (w * (dilation + 1)) - dilation))
        for i in range(h):
            for j  in range(w):
                kernel_out[i * (dilation + 1)][j * (dilation + 1)] = kernel[i][j]

        # Return dilated kernel
        return kernel_out
    

    # Made with assistance from https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    # Made with assistance from Kingma & Lei Ba (2015) - https://arxiv.org/pdf/1412.6980
    def update_layer(
            self, 
            weights_grad: np.ndarray, 
            bias_grad: np.ndarray, 
            lr: float, 
            epoch_num: int, 
            clip_value=1.0, 
            beta_1=None, 
            beta_2=None,
            l2_lambda=0
        ):
        # Check dimensionality of update values provided
        if not weights_grad.shape == self._kernel.shape:
            raise ValueError("Error: kernel gradients do not match dimensionality of kernels array")
        
        # Find the greatest gradient norm
        norm = np.linalg.norm(weights_grad)

        # Apply clipping
        if norm > clip_value:
            weights_grad = weights_grad * (clip_value / norm)
            bias_grad = bias_grad * (clip_value / norm)

        # Initialize the values to change weights and bias by
        dW = None
        dB = None

        # Get values to update weights and bias by
        if not beta_1 is None:
            # Use Adam
            self.update_velocity(weights_grad, bias_grad, beta_1, beta_2, epoch_num)
            dW = -(lr * (self._m_hat_w / (np.sqrt(self._v_hat_w) + 1e-8)) + lr * l2_lambda * self._kernel)
            dB = -(lr * (self._m_hat_b / (np.sqrt(self._v_hat_b) + 1e-8)) + lr * l2_lambda * self._bias)
        else:
            # Do not apply momentum
            dW = -lr * weights_grad
            dB = -lr * bias_grad

        # Update layer
        self._kernel += dW
        self._bias += dB

    def _he_initialize_kernel(self, kernel_size: int):
        fan_in = kernel_size * kernel_size
        kernel = np.random.normal(0, np.sqrt(2 / fan_in), (kernel_size, kernel_size))
        return kernel
    
    def _initialize_biases(self):
        biases = np.random.normal(0, 1) / 100
        return biases
    
    def __str__(self):
        to_ret = f'{self._K}x{self._K} kernel'
        to_ret += f'\n{self._S} stride, {self._P} padding'
        to_ret += f'\nInput size: {self._input_dims}'
        to_ret += f'\nActivation function: {self._activation_fn}'
        return to_ret


class RegularConvolutionalLayer(ConvolutionalLayer):
    # CONSTRUCTOR
    def __init__(
            self,
            kernel_size:int=3,
            stride:int=1,
            padding:int=0,
            activation_fn=None,
            activation_fn_dx=None,
            input_dims:tuple=(28, 28),
        ):
        super().__init__(kernel_size, stride, padding, activation_fn, activation_fn_dx, input_dims)

    # GETTER METHODS
    def get_output_size(self):
        h, w = self._input_dims
        K = self._K
        p = self._P
        s = self._S
        return (int((h - K + 2 * p) / s) + 1, int((w - K + 2 * p) / s) + 1)

    # FUNCTIONAL METHODS
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Check input is valid
        if not len(x) == self._input_dims[0] and len(x[0]) == self._input_dims[1]:
            raise ValueError(f"Error: input of invalid dimensionality was provided. Shape {self._input_dims} is required")
        self._input = x.copy()
        
        # Apply convolution
        self._z_values = ConvolutionalLayer.convolve(x, self._kernel, self._bias, self._P, self._S)
        
        # Return value
        self._output = self._activation_fn(self._z_values.copy())
        return self._output
    
    def __str__(self):
        to_ret = 'CONVOLUTIONAL LAYER\n'
        to_ret += super().__str__()
        return to_ret



class TransposedConvolutionalLayer(ConvolutionalLayer):
    # CONSTRUCTOR
    def __init__(
        self,
        kernel_size:int=3,
        stride:int=1,
        padding:int=0,
        activation_fn=None,
        activation_fn_dx=None,
        input_dims:tuple=(4, 4),
    ):
        super().__init__(kernel_size, stride, padding, activation_fn, activation_fn_dx, input_dims)

    # GETTER METHODS
    def get_output_size(self):
        h, w = self._input_dims
        return ((h - 1) * self._S + self._K, (w - 1) * self._S + self._K)

    # FUNCTIONAL METHODS
    # Made with assistance from https://medium.com/data-science/understand-transposed-convolutions-and-build-your-own-transposed-convolution-layer-from-scratch-4f5d97b2967
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Check input is valid
        if not len(x) == self._input_dims[0] and len(x[0]) == self._input_dims[1]:
            raise ValueError(f"Error: input of invalid dimensionality was provided. Shape {self._input_dims} is required")
        self._input = x.copy()
        
        # Apply padding
        h, w = x.shape # number of rows, number of columns, number of channels

        # Initialize input and output variables
        self._z_values = np.zeros(((h - 1) * self._S + self._K, (w - 1) * self._S + self._K))
        # out_h, out_w= self._z_values.shape

        # Perform tranposed convolution
        for row in range(h):
            for col in range(w):
                # Get block in z values to output transposed convolution to
                start_row = row * self._S
                end_row = start_row + self._K
                start_col = col * self._S
                end_col = start_col + self._K

                # Get the values
                val = x[row][col]
                vals = np.multiply(val, self._kernel)
                self._z_values[start_row:end_row, start_col:end_col] = self._z_values[start_row:end_row, start_col:end_col] + vals

        # Add bias
        self._z_values += np.ones(self._z_values.shape) * self._bias

        # Return value
        self._output = self._activation_fn(self._z_values.copy())
        return self._output
    
    def __str__(self):
        to_ret = 'TRANSPOSED CONVOLUTIONAL LAYER\n'
        to_ret += super().__str__()
        return to_ret



# The code for this class is adapted from my HYP
class FullyConnectedLayer(Layer):

    def __init__(
        self,
        weights:np.ndarray=None,
        bias:np.ndarray=None,
        activation_fn=None,
        activation_fn_dx=None,
        num_inputs=8,
        num_outputs=4
    ):
        # Initialize superclass
        super().__init__(activation_fn, activation_fn_dx)

        # Initialize forwarding properties
        self._weights = weights
        self._bias = bias

        # Set properties for dimensions
        self._input_size = None
        self._output_size = None
        if not self._weights is None:
            self._input_size = self._weights.shape[1] # number of inputs is number of columns
            self._output_size = self._weights.shape[0] # number of outputs is number of rows 
            if self._bias is None:
                raise ValueError("Error: weights provided without bias vector")
        else:
            self._input_size = num_inputs
            self._output_size = num_outputs
            self._weights = self._he_initialize_weights(num_inputs, num_outputs)
            self._bias = np.random.uniform(0, 1, num_outputs)

        # Create inputs and outputs
        if not self._input_size is None:
            self._input = np.zeros(self._input_size)
            self._output = np.zeros(self._output_size)

    # SETTER METHODS
    def set_weights(self, weights:np.ndarray):
        # Check dimensions of inputted weights if self._weights is already set
        if not self._weights is None:
            if not self._weights.shape == weights.shape:
                raise ValueError("Error: dimensions of current weights do not match dimensions of inputted weights")
            
        # Check if there are any NaNs in inputted weights
        if np.any(np.isnan(weights)):
            raise ValueError("Error: inputted weights contain at least one NaN value")
        
        # Set weights
        self._weights = weights

    def set_biases(self, bias:np.ndarray):
        # Check dimensions if self._biases is already set
        if not self._bias is None:
            if not self._bias.shape == bias.shape:
                raise ValueError("Error: length of current bias vector does not match length of inputted bias vector")
            
        # Check for NaN values
        if np.any(np.isnan(bias)):
            raise ValueError("Error: inputted bias vector contains at least one NaN value")
        
        # Set bias
        self._bias = bias

    # GETTER METHODS
    def get_weights(self) -> np.ndarray:
        return self._weights
    
    def get_biases(self) -> np.ndarray:
        return self._bias

    def get_z_values(self) -> np.ndarray:
        return self._z_values
    
    def get_activation_function_dx(self):
        return self._activation_fn(self._z_values)
    
    def get_input_size(self):
        return self._input_size
    
    def get_output_size(self):
        return self._output_size
    
    def contains_nan(self):
        return np.any(np.isnan(self._weights))

    # FUNCTIONAL METHODS
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Set input value
        self._input = x.copy()

        # Get z values
        self._z_values = np.add(np.dot(self._weights, self._input), self._bias)

        # Apply activation function and get 
        self._output = self._activation_fn(self._z_values.copy())
        return self._output
    
    # Apply updates to layer
    # Made with assistance from https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    # Made with assistance from Kingma & Lei Ba (2015) - https://arxiv.org/pdf/1412.6980
    def update_layer(
            self, 
            weights_grad: np.ndarray, 
            bias_grad: np.ndarray, 
            lr: float, 
            epoch_num: int, 
            clip_value=1.0, 
            beta_1=None, 
            beta_2=None,
            l2_lambda=0
        ):
        # Apply clipping
        weights_grad_norm = np.linalg.norm(weights_grad)
        if weights_grad_norm > clip_value:
            weights_grad = weights_grad * (clip_value / float(weights_grad_norm))
            bias_grad = bias_grad * (clip_value / float(weights_grad_norm))

        # Initialize the values to change weights and bias by
        dW = None
        dB = None

        # Get values to update weights and bias by
        if not beta_1 is None:
            # Use Adam
            self.update_velocity(weights_grad, bias_grad, beta_1, beta_2, epoch_num)
            dW = -(lr * (self._m_hat_w / (np.sqrt(self._v_hat_w) + 1e-8)) + lr * l2_lambda * self._weights) 
            dB = -(lr * (self._m_hat_b / (np.sqrt(self._v_hat_b) + 1e-8)) + lr * l2_lambda * self._bias)
        else:
            # Do not apply momentum
            dW = -lr * weights_grad
            dB = -lr * bias_grad

        # Update layer
        self._weights += dW
        self._bias += dB

    def _he_initialize_weights(self, n_inputs, n_outputs):
        return np.random.normal(0, np.sqrt(2 / n_inputs), (n_outputs, n_inputs))
    
    def __str__(self):
        to_ret = 'FULLY CONNECTED LAYER'
        to_ret += f'{self._input_size} inputs'
        to_ret += f'\n{self._output_size} outputs'
        to_ret += f'\nActivation function: {self._activation_fn}'
        return to_ret


class SplitHeadFullyConnectedLayer(Layer):
    # CONSTRUCTOR
    def __init__(
        self,
        mean_layer:FullyConnectedLayer,
        log_var_layer:FullyConnectedLayer
    ):
        # Check that layers have same output size
        if mean_layer.get_output().shape != log_var_layer.get_output().shape:
            raise ValueError("Error: dimensions of the outputs of the layers provided do not match, meaning VAE is invalid")
        
        # Initialize super class
        super().__init__(None, None)

        # Initialize layers
        self._mean_layer = mean_layer
        self._log_var_layer = log_var_layer

        # Initialize input and output values
        self._input = None
        self._output = None

        # Mean, log variance, and epsilon values
        self._mean = None
        self._log_var = None
        self._epsilon = None


    # GETTER METHODS
    def get_mean_weights(self) -> np.ndarray:
        return self._mean_layer.get_weights()
    
    def get_log_var_weights(self) -> np.ndarray:
        return self._log_var_layer.get_weights()

    def get_biases(self, get_mean_layer:bool=True) -> np.ndarray:
        if get_mean_layer:
            return self._mean_layer.get_biases()
        else:
            return self._log_var_layer.get_biases()

    def get_mean(self):
        return self._mean
    
    def get_log_var(self):
        return self._log_var
    
    def get_epsilon(self):
        return self._epsilon
    
    def get_activation_function_dx(self, get_mean_layer:bool=True):
        if get_mean_layer:
            return self._mean_layer.get_activation_function_dx()
        else:
            return self._log_var_layer.get_activation_function_dx()
        
    def get_input_size(self):
        return self._mean_layer.get_input_size()
    
    def get_output_size(self):
        return self._mean_layer.get_output_size()
    
    def apply_activation_fn_dx(self):
        return 1
    
    def contains_nan(self):
        return self._mean_layer.contains_nan() or self._log_var_layer.contains_nan()
        
    # FUNCTIONAL METHODS
    def forward(self, x:np.ndarray):
        # Get z value sampled from latent space
        self._input = x.copy()
        self._mean = self._mean_layer.forward(x)
        self._log_var = self._log_var_layer.forward(x)
        self._epsilon = np.random.normal(0, 1, len(self._mean))
        self._output = self._mean + np.exp(self._log_var * 0.5) * self._epsilon

        # Return these values
        return self._output
    
    def update_layer(
            self, 
            weights_grad: tuple[np.ndarray, np.ndarray], 
            bias_grad: tuple[np.ndarray, np.ndarray], 
            lr: float, 
            epoch_num: int, 
            clip_value=1.0, 
            beta_1=None, 
            beta_2=None,
            l2_lambda=0
        ):
        # Get the weights for each head
        mean_weights_grad = weights_grad[0]
        log_var_weights_grad = weights_grad[1]
        mean_bias_grad = bias_grad[0]
        log_var_bias_grad = bias_grad[1]

        # Perform gradient clipping
        mean_weights_grad_norm = np.linalg.norm(mean_weights_grad)
        log_var_weights_grad_norm = np.linalg.norm(log_var_weights_grad)
        if mean_weights_grad_norm > clip_value or log_var_weights_grad_norm > clip_value:
            # Get the larger of the 2 norms
            max_norm = mean_weights_grad_norm if mean_weights_grad_norm > log_var_weights_grad_norm else log_var_weights_grad_norm

            # Reduce to be within the clip value for all weights and biases
            mean_weights_grad = mean_weights_grad * (clip_value / float(max_norm))
            mean_bias_grad = mean_bias_grad * (clip_value / float(max_norm))
            log_var_weights_grad = log_var_weights_grad * (clip_value / float(max_norm))
            log_var_bias_grad = log_var_bias_grad * (clip_value / float(max_norm))

        # Update each layer
        self._mean_layer.update_layer(
            weights_grad=mean_weights_grad, 
            bias_grad=mean_bias_grad, 
            lr=lr, 
            epoch_num=epoch_num,
            clip_value=clip_value, 
            beta_1=beta_1,
            beta_2=beta_2,
            l2_lambda=l2_lambda
        )
        self._log_var_layer.update_layer(
            weights_grad=log_var_weights_grad, 
            bias_grad=log_var_bias_grad, 
            lr=lr, 
            epoch_num=epoch_num,
            clip_value=clip_value, 
            beta_1=beta_1,
            beta_2=beta_2,
            l2_lambda=l2_lambda
        )

    def __str__(self):
        to_ret = 'SPLIT HEAD LAYER'
        to_ret += '\n-----------------------------------------------------'
        to_ret += '\nMEAN LAYER:'
        to_ret += '\n' + self._mean_layer.__str__()
        to_ret += '\n-----------------------------------------------------'
        to_ret += '\nLOG VARIANCE LAYER:'
        to_ret += '\n' + self._log_var_layer.__str__()
        to_ret += '\n-----------------------------------------------------'
        to_ret += f'\nLatent space size: {self._mean_layer.get_output_size()}'
        return to_ret