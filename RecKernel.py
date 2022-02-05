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
                 res_scale=1, input_scale=1, bias_scale=0, 
                 memory_efficient=True, n_iter=50, step=1, device=device):
        self.function = function
        self.res_scale = res_scale
        self.input_scale = input_scale
        self.device = device

    def forward(self, input_data, initial_K=None, gram_dim=2):
        """
        Computes the forward pass of a recurrent kernel.
        Parameters:
        - input_data: data of shape (input_len, input_dim) or (n_input, n_iter, input_dim)
        - initial_K: initial value of the kernel matrix if provided

        Returns:
        - K: kernel matrix of size (n_input, n_input)
        - diag_res: if return_diag==True, returns this diagonal matrix for later uses
        """
        input_len, input_dim = input_data.shape
        n_iter = self.n_iter

        if initial_K is None:
            K = torch.eye((gram_dim, gram_dim)).to(self.device)
        else:
            K = initial_K

        for t in range(n_iter):
            current_input = input_data[t, :]
            input_norm = current_input @ current_input.T
            input_gram = input_norm.repeat(gram_dim, gram_dim)

            K = self.update_kernel(K, input_gram)
        return K

    def update_kernel(self, K, input_gram):
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
        elif self.function == 'acos relu':
            diag_res = torch.diag(K)
            diag_in = torch.diag(input_gram)
            renorm_diag = 1 / (
                self.res_scale**2 * diag_res + self.input_scale**2 * diag_in)
            renorm_factor = torch.sqrt(
                renorm_diag.reshape(gram_dim, 1) @ renorm_diag.reshape(1, gram_dim))
            correl = self.res_scale**2 *K + self.input_scale**2 *input_gram
            correl_norm = correl * renorm_factor
            K = (1/20*np.pi) / renorm_factor * (correl_norm * torch.acos(-correl_norm) + torch.sqrt(1-correl_norm**2))
        return K

    def stability_test(self, input_data):        
        initial_state_1 = torch.randn(self.res_size) / np.sqrt(self.res_size)
        initial_state_2 = torch.randn(self.res_size) / np.sqrt(self.res_size)
        trained_state_1 = self.forward(input_data,initial_state_1)
        trained_state_2 = self.forward(input_data,initial_state_2)
        state_diff = torch.linalg.norm(trained_state_1-trained_state_2, dim=1)

    def test_stability_upt(self,input_data,initial_G_eq = 1,initial_G_ineq = 0):
        G_eq = initial_G_eq
        G_ineq = initial_G_ineq

        n_iter = input_data.shape[0]

        res = torch.ones(n_iter+1)*2
        for t in range(n_iter):
            if self.function == 'arcsin':
                G_eq = 2 / np.pi * np.arcsin(
                     (2* self.res_scale**2 * G_eq + 2* self.input_scale ** 2) / 
                     (1+ 2* self.res_scale**2 * G_eq + 2* self.input_scale ** 2))
                G_ineq = 2 / np.pi * np.arcsin(
                     (2* self.res_scale**2 * G_ineq + 2* self.input_scale ** 2) / 
                     (1+ 2* self.res_scale**2 * G_eq + 2* self.input_scale ** 2))
            elif self.function == 'heaviside':
                G_eq   = 1/2
                G_ineq = 1/2 - 1/2/np.pi * np.arccos(
                    (self.res_scale ** 2 * G_ineq + self.input_scale ** 2) /
                    (self.res_scale ** 2 * G_eq + self.input_scale ** 2) )
            elif self.function == 'relu':
                G_eq = 1 / 2 * (self.res_scale ** 2 * G_eq + self.input_scale ** 2)
                G_ineq = 1 / 2 /np.pi * (self.res_scale ** 2 * G_ineq + self.input_scale ** 2) * np.arccos(
                    -(self.res_scale ** 2 * G_ineq + self.input_scale ** 2) /
                     (self.res_scale ** 2 * G_eq + self.input_scale ** 2 + 1e-5)) + 1 / 2 / np.pi * np.sqrt(
                    (self.res_scale ** 2 * G_eq + self.input_scale ** 2)**2 - (self.res_scale ** 2 * G_ineq + self.input_scale ** 2)**2)

            res[t+1] = G_eq-G_ineq

        return res, G_eq, G_ineq