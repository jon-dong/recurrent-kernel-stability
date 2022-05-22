import numpy as np
import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class RecKernel():
    """
    Implements a recurrent kernel for stability tests.
    Parameters:
    - function: kernel function of the recurrent kernel
    - res_scale: scale of the corresponding reservoir weights
    - input_scale: scale of the corresponding input-to-reservoir weights
    """
    
    def __init__(self, function='arcsin', 
                 res_scale=1, input_scale=1, 
                 device=device):
        self.function = function
        self.res_scale = res_scale
        self.input_scale = input_scale
        self.device = device

    def forward(self, input_data, initial_K=None, gram_dim=2):
        """
        Computes the forward pass of a recurrent kernel. For simplicity and conciseness, this function only implements the forward pass for a shared input.
        Parameters:
        - input_data: data of shape (input_len, input_dim)
        - initial_K: initial value of the kernel matrix if provided
        - gram_dim: size of the Gram matrix

        Returns:
        - K: kernel matrix of size (gram_dim, gram_dim)
        """
        input_len = input_data.shape[0]
        if initial_K is None:
            K = torch.eye(gram_dim, gram_dim).to(self.device)
        else:
            K = initial_K
            gram_dim = K.shape[0]

        for t in range(input_len):
            current_input = input_data[t, :]
            input_norm = current_input @ current_input.T
            input_gram = input_norm.repeat(gram_dim, gram_dim)
            # Gram matrix for a shared input

            K = self.update_kernel(K, input_gram)
        return K

    def forward_general(self, input_data, initial_K=None):
        """
        Computes the forward pass of a recurrent kernel. This function implements the forward pass for a set of inputs, in a more general case for convergence tests.
        Parameters:
        - input_data: data of shape (n_input, input_len, input_dim)
        - initial_K: initial value of the kernel matrix if provided
        - gram_dim: size of the Gram matrix

        Returns:
        - gram_concat: kernel matrix of size (input_len+1, gram_dim, gram_dim)
        """
        gram_dim, input_len, input_dim = input_data.shape
        if initial_K is None:
            K = torch.eye(gram_dim, gram_dim).to(self.device)
        else:
            K = initial_K

        gram_concat = torch.zeros(input_len+1, gram_dim, gram_dim)
        gram_concat[0, :, :] = K
        for t in range(input_len):
            current_input = input_data[:, t, :]
            input_gram = current_input @ current_input.T
            K = self.update_kernel(K, input_gram, t=t)
            gram_concat[t+1, :, :] = K
        return gram_concat

    def update_kernel(self, K, input_gram, t=-1):
        """
        Computes one iteration of the Recurrent Kernel. 
        Parameters:
        - K: current Gram matrix of size (gram_dim, gram_dim)
        - input_gram: current input Gram matrix of size (gram_dim, gram_dim)

        Returns:
        - K: next Gram matrix of size (gram_dim, gram_dim)
        """
        gram_dim = K.shape[0]
        if self.function == 'linear':
            K = self.res_scale**2 * K + self.input_scale**2 * input_gram
        elif self.function == 'arcsin':
            diag_res = torch.diag(K)
            diag_in = torch.diag(input_gram)    
            renorm_diag = 1 / (
                1 + 2 * self.res_scale**2 * diag_res + 2 * self.input_scale**2 * diag_in)
            renorm_factor = torch.sqrt(
                renorm_diag.reshape(gram_dim, 1) @ renorm_diag.reshape(1, gram_dim))
            K = 2 / np.pi * torch.asin(
                (2 * self.res_scale**2 * K + 2 * self.input_scale**2 * input_gram) * renorm_factor)
        elif self.function == 'acos heaviside':
            diag_res = torch.diag(K)
            diag_in = torch.diag(input_gram)
            renorm_diag = 1 / (
                self.res_scale**2 * diag_res + self.input_scale**2 * diag_in)
            renorm_factor = torch.sqrt(
                renorm_diag.reshape(gram_dim, 1) @ renorm_diag.reshape(1, gram_dim))
            K = 0.5 - torch.acos(
                (self.res_scale**2 * K + self.input_scale**2 *input_gram) * renorm_factor) /(2*np.pi) 
        elif self.function == 'asin sign':
            diag_res = torch.diag(K)
            diag_in = torch.diag(input_gram)
            renorm_diag = 1 / (
                self.res_scale**2 * diag_res + self.input_scale**2 * diag_in)
            renorm_factor = torch.sqrt(
                renorm_diag.reshape(gram_dim, 1) @ renorm_diag.reshape(1, gram_dim))
            K = 2 / np.pi * torch.asin(
                (self.res_scale**2 * K + self.input_scale**2 *input_gram) * renorm_factor)
            # if t==0:
            #     print(K)
        elif self.function == 'acos relu':
            diag_res = torch.diag(K)
            diag_in = torch.diag(input_gram)
            renorm_diag = self.res_scale**2 * diag_res + self.input_scale**2 * diag_in
            renorm_factor = torch.sqrt(
                renorm_diag.reshape(gram_dim, 1) @ renorm_diag.reshape(1, gram_dim))
            correl = self.res_scale**2 *K + self.input_scale**2 *input_gram
            correl_norm = correl / (renorm_factor + 1e-5)
            K = (1/2/np.pi) * renorm_factor * (correl_norm * torch.acos(-correl_norm) + torch.sqrt(1-correl_norm**2))
        return K

    def stability_test(self, input_data):
        input_len = input_data.shape[0]
        gram_dim = 2
        K = torch.eye(gram_dim, gram_dim).to(self.device)
        
        stability_metric = torch.ones(input_len+1) * 2
        for t in range(input_len):
            current_input = input_data[t, :]
            input_norm = current_input @ current_input.T
            input_gram = input_norm.repeat(gram_dim, gram_dim)
            K = self.update_kernel(K, input_gram)
            stability_metric[t+1] = 2 * (K[0, 0] - K[0, 1])

        return stability_metric
    
    