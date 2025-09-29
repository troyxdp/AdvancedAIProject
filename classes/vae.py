import pickle

import numpy as np

from classes.layer import *

# Adapted from my HYP
class VariationalAutoencoder(): 

    def __init__(
        self, 
        input_size:int=28,
        output_size:int=28,
        layers:list[Layer]=None
    ):
        self._input = np.zeros(input_size, dtype=np.float64)
        self._layers: list[Layer] = layers if not layers is None else []
        self._output = np.ones(output_size, dtype=np.float64)

        self._mean = None
        self._log_var = None

    # ACTIVATION FUNCTIONS
    def relu(x: np.ndarray):
        x_flat = x.flatten()
        to_ret = []
        for val in x_flat:
            if val > 0:
                to_ret.append(val)
            else:
                to_ret.append(0)
        return np.array(to_ret, dtype=np.float64).reshape(x.shape)
    def relu_dx(x: np.ndarray):
        x_flat = x.flatten()
        to_ret = []
        for val in x_flat:
            if val > 0:
                to_ret.append(1)
            else:
                to_ret.append(0)
        return np.array(to_ret, dtype=np.float64).reshape(x.shape)
    def sigmoid(x: np.ndarray):
        return 1/(1 + np.exp(-x))
    def sigmoid_dx(x: np.ndarray):
        return VariationalAutoencoder.sigmoid(x) * (1 - VariationalAutoencoder.sigmoid(x))
    def linear(x: np.ndarray):
        return x
    def linear_dx(x: np.ndarray):
        return np.ones(x.shape)


    # GETTER METHODS
    def get_num_layers(self):
        return len(self._layers)
    
    def get_num_inputs(self):
        return len(self._input)

    def get_layer(self, layer_num):
        return self._layers[layer_num]
        
    def get_output(self):
        return self._output


    # SETTER METHODS
    def append_layer(self, layer: Layer):
        layer_input_size = 0
        last_layer_output_size = 0
        if not len(self._layers) == 0:
            # Get input size to layer and input size of layer
            if isinstance(layer, FullyConnectedLayer) or isinstance(layer, SplitHeadFullyConnectedLayer):
                layer_input_size = layer.get_input_size()
                if isinstance(self._layers[-1], RegularConvolutionalLayer) or isinstance(self._layers[-1], TransposedConvolutionalLayer):
                    last_layer_output_size = self._layers[-1].get_output_size()
                    last_layer_output_size = last_layer_output_size[0] * last_layer_output_size[1] # flatten
                else:
                    last_layer_output_size = self._layers[-1].get_output_size()
            else:
                layer_input_size = layer.get_input_size()
                if isinstance(self._layers[-1], FullyConnectedLayer) or isinstance(self._layers[-1], SplitHeadFullyConnectedLayer):
                    layer_input_size = layer_input_size[0] * layer_input_size[1] # flatten
                    last_layer_output_size = self._layers[-1].get_output_size()
                else:
                    last_layer_output_size = self._layers[-1].get_output_size()
            
            # Check if the input size to the layer is equal to the input size of the layer
            if not layer_input_size == last_layer_output_size:
                raise ValueError("Error: input size of layer and size of input provided to layer do not match")

        # Append layer
        self._layers.append(layer)

    # FUNCTIONAL METHODS
    def forward(self, x:np.ndarray):
        self._input = x.copy()
        for i, layer in enumerate(self._layers):
            # Do reshaping of input
            if isinstance(layer, FullyConnectedLayer):
                if isinstance(self._layers[i-1], RegularConvolutionalLayer):
                    x = x.flatten()
            if isinstance(layer, TransposedConvolutionalLayer):
                if isinstance(self._layers[i-1], FullyConnectedLayer):
                    x = x.reshape( ( int(np.sqrt(len(x))) , int(np.sqrt(len(x))) ) )
            
            # Forward the input
            x = layer.forward(x)
        self._output = x
        return self._output

    def back_propogate(self, lr: float, error_prime: np.ndarray,  momentum=None, clip_score=1.0, l2_lambda=0):
        # Get first delta value
        delta: np.ndarray = error_prime * self._layers[-1].apply_activation_fn_dx() # multiply gradient of cost function with derivative of activation function applied to z values
        
        # Do backprop for final layer, which is a transposed convolutional layer
        bias_grad = np.sum(delta)
        weights_grad = ConvolutionalLayer.convolve(delta, self._layers[-1].get_input())
        self._layers[-1].update_layer(weights_grad, bias_grad, lr, clip_score, momentum)

        # Propogate through the other layers from the 2nd last hidden layer to the 1st hidden layer at index 0 of self._layers
        # . is dot product, * is element-wise (hadamard) product, x is convolution
        for l in range(len(self._layers) - 2, -1, -1):
            # Get each of the different layers
            curr_layer: Layer = self._layers[l]
            next_layer: Layer = self._layers[l+1]

            # Initialize values
            weights_grad = None
            bias_grad = None

            # Get s_l'(z_l) where s_l is activation function of layer l, the current layer
            act_fn_dx = curr_layer.apply_activation_fn_dx()

            # Perform unique logic for each type of layer
            if isinstance(curr_layer, FullyConnectedLayer):
                # Can be followed by a transposed convolutional layer in the decoder or a split head fully connected layer in the encoder
                if isinstance(next_layer, FullyConnectedLayer):
                    # Get delta_l = dC/dZ_l = ((W_l+1)^T . delta_(l+1)) * s_l'(z_l) where W_l+1 is weights of next layer 
                    delta = np.dot(next_layer.get_weights().transpose(), delta) * act_fn_dx
                elif isinstance(next_layer, TransposedConvolutionalLayer):
                    # Get delta_l = delta_(l+1) x W_rotated_(l) * s_l'(z_l)
                    delta = ConvolutionalLayer.convolve(delta, next_layer.get_rotated_kernel()).flatten() * act_fn_dx
                elif isinstance(next_layer, SplitHeadFullyConnectedLayer):
                    # Get delta_l = ((W^mu_l+1)^T . delta^mu_l+1 + (W^sigma_l+1)^T . delta^sigma_l+1) . s_l'(z_l)
                    delta = (np.dot(next_layer.get_mean_weights().transpose(), delta[0]) + np.dot(next_layer.get_log_var_weights().transpose(), delta[1])) * act_fn_dx
                else:
                    raise Exception("Error: can only have a fully connected layer followed by a transposed convolutional layer or a split head fully convolutional layer in a VAE")
                
                # Get the weights grad and bias grad
                # Weights gradient is delta . a_l-1 where a_l-1 is the activated output of the previous layer/input of current layer
                weights_grad = np.outer(delta, curr_layer.get_input())
                bias_grad = delta

            elif isinstance(curr_layer, RegularConvolutionalLayer):
                # Can be followed by a fully connected layer at the end of the encoder or another regular convolutional layer in the encoder
                if isinstance(next_layer, FullyConnectedLayer):
                    delta = np.reshape(np.dot(next_layer.get_weights().transpose(), delta), curr_layer.get_output_shape()) * act_fn_dx
                elif isinstance(next_layer, RegularConvolutionalLayer):
                    # delta_l = delta_l+1 x W_rotated_(l+1) *f s_l'(z_l) where *f is full mode convolution
                    delta = ConvolutionalLayer.convolve(x=delta, kernel=next_layer.get_rotated_kernel(), bias=None, p=next_layer.get_kernel_shape()[0]-1) * act_fn_dx
                else:
                    raise Exception("Error: can only have a regular convolution layer followed by a fully connected layer or another regular convolution layer in a VAE")
                
                # Get the delta value, weights grad, and bias grad
                weights_grad = ConvolutionalLayer.convolve(curr_layer.get_input(), delta)
                bias_grad = np.sum(delta)

            elif isinstance(curr_layer, TransposedConvolutionalLayer):
                # Can only be followed by another transposed convolutional layer in the decoder
                # Get the delta value, weights grad, and bias grad
                delta = ConvolutionalLayer.convolve(x=delta, kernel=next_layer.get_rotated_kernel(), s=next_layer.get_stride()) * act_fn_dx
                weights_grad = ConvolutionalLayer.convolve(delta, ConvolutionalLayer.dilate(curr_layer.get_input(), curr_layer.get_stride() - 1))
                bias_grad = np.sum(delta) # bias grad is the sum of all the delta values. Can show this quite easily

            elif isinstance(curr_layer, SplitHeadFullyConnectedLayer):
                # Can only be followed by a fully connected layer in the decoder
                # Get the delta value
                delta_z = np.dot(next_layer.get_weights().transpose(), delta)
                delta_mu = delta_z + curr_layer.get_mean()
                delta_sigma = 0.5 * (delta_z * curr_layer.get_epsilon() * np.exp(0.5 * curr_layer.get_log_var()) + np.exp(curr_layer.get_log_var()) - 1)
                delta = (delta_mu, delta_sigma)

                # Get the weights and bias grads
                weights_grad_mean = np.outer(delta[0], curr_layer.get_input())
                bias_grad_mean = delta[0]
                weights_grad_log_var = np.outer(delta[1], curr_layer.get_input())
                bias_grad_log_var = delta[1]

                # Combine these into tuples for passing to update_layer()
                weights_grad = (weights_grad_mean, weights_grad_log_var)
                bias_grad = (bias_grad_mean, bias_grad_log_var)
            
            # Update the layer
            curr_layer.update_layer(weights_grad, bias_grad, lr, clip_score, momentum)

    def save_network(self, file_path: str):
        # Check that there are no NaN values in any of the layers
        for layer in self._layers:
            if layer.contains_nan():
                raise ValueError("Error: layer contains at least one NaN value")
            
        # Pickle network
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def load_network(file_path: str):
        # Load object
        to_ret = None
        with open(file_path, 'rb') as f:
            to_ret = pickle.load(f)

        # Check network is valid
        for i in range(len(to_ret._layers) - 1):
            curr_layer = to_ret.get_layer(i)
            next_layer = to_ret.get_layer(i + 1)
            if isinstance(curr_layer, FullyConnectedLayer) and isinstance(next_layer,  FullyConnectedLayer) and curr_layer._output_size != next_layer._input_size:
                raise ValueError("Error: invalid network provided")
            
        # Return network
        return to_ret
    
    def __str__(self):
        to_ret = '..................................................................'
        to_ret += '\n=================================================================='
        for i, layer in enumerate(self._layers):
            to_ret += f"\nLAYER {i + 1}:\n"
            to_ret += str(layer)
            to_ret += '\n=================================================================='
        to_ret += '\n..................................................................'
        return to_ret

    def __repr__(self):
        return self.__str__()
    


if __name__ == '__main__':
    # Instantiate VAE
    vae = VariationalAutoencoder()

    # ENCODER
    # Layer 1
    layer_1 = RegularConvolutionalLayer(
        kernel_size=3,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(28, 28)
    )
    vae.append_layer(layer_1)
    # Layer 2
    layer_2 = RegularConvolutionalLayer(
        kernel_size=3,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(26, 26)
    )
    vae.append_layer(layer_2)
    # Layer 3
    layer_3 = RegularConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(24, 24)
    )
    vae.append_layer(layer_3)
    # Layer 4
    layer_4 = RegularConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(20, 20)
    )
    vae.append_layer(layer_4)
    # Layer 5
    layer_5 = RegularConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(16, 16)
    )
    vae.append_layer(layer_5)
    # Layer 7
    layer_7 = FullyConnectedLayer(
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        num_inputs=12*12,
        num_outputs=8*8
    )
    vae.append_layer(layer_7)
    # Layer 8
    mean_layer = FullyConnectedLayer(
        activation_fn=VariationalAutoencoder.linear,
        activation_fn_dx=VariationalAutoencoder.linear_dx,
        num_inputs=8*8,
        num_outputs=4*4
    )
    log_var_layer = FullyConnectedLayer(
        activation_fn=VariationalAutoencoder.linear,
        activation_fn_dx=VariationalAutoencoder.linear_dx,
        num_inputs=8*8,
        num_outputs=4*4
    )
    layer_8 = SplitHeadFullyConnectedLayer(
        mean_layer=mean_layer,
        log_var_layer=log_var_layer
    )
    vae.append_layer(layer_8)

    # DECODER
    # Layer 9
    layer_9 = FullyConnectedLayer(
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        num_inputs=4*4,
        num_outputs=8*8,
    )
    vae.append_layer(layer_9)
    # Layer 10
    layer_10 = TransposedConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu_dx,
        input_dims=(8, 8)
    )
    vae.append_layer(layer_10)
    # Layer 11
    layer_11 = TransposedConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu,
        input_dims=(12, 12)
    )
    vae.append_layer(layer_11)
    # Layer 12
    layer_12 = TransposedConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu,
        input_dims=(16, 16)
    )
    vae.append_layer(layer_12)
    # Layer 13
    layer_13 = TransposedConvolutionalLayer(
        kernel_size=5,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu,
        input_dims=(20, 20)
    )
    vae.append_layer(layer_13)
    # Layer 14
    layer_14 = TransposedConvolutionalLayer(
        kernel_size=3,
        activation_fn=VariationalAutoencoder.relu,
        activation_fn_dx=VariationalAutoencoder.relu,
        input_dims=(24, 24)
    )
    vae.append_layer(layer_14)
    # Layer 15
    layer_15 = TransposedConvolutionalLayer(
        kernel_size=3,
        activation_fn=VariationalAutoencoder.linear,
        activation_fn_dx=VariationalAutoencoder.linear,
        input_dims=(26, 26)
    )
    vae.append_layer(layer_15)
    print(vae)


    # Test backprop
    print()
    print("Error from array of 1's:")
    input = np.random.normal(1, 1, (28, 28))
    output = vae.forward(input)
    target = np.ones_like(output)
    for i in range(2000):
        output = vae.forward(input)
        error = np.dot(np.subtract(target, output).flatten(), np.subtract(target, output).flatten())
        if i % 50 == 0:
            print(error)

        error_prime = output - target
        vae.back_propogate(
            lr=0.0001,
            error_prime=error_prime,
            clip_score = 1.0,
            # momentum=0.1
        )
    print("\nSample output:")
    print(output)

    # Test saving
    print("\nSaving network...")
    vae.save_network('/home/troyxdp/Documents/University Work/Advanced Artificial Intelligence/Project/networks/demo.pkl')

    # Test loading
    print("\nLoading network...")
    vae_1 = VariationalAutoencoder.load_network('/home/troyxdp/Documents/University Work/Advanced Artificial Intelligence/Project/networks/demo.pkl')
    print(vae_1)
    print("\nDemo output:")
    print(vae_1.forward(input))