#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch

from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal


class hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, dist_class, estimate_batch, device):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion
        self.dist_class = dist_class
        if isinstance(estimate_batch, MultiAgentBatch):
            self.estimate_batch = estimate_batch.as_sample_batch()
        else:
            self.estimate_batch = estimate_batch
        self.estimate_batch = self.estimate_batch.to_device(device)
        self.device = device

        # pre-processing for single batch case to simplify the computation.
        # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
        loss = self.criterion(self.model, self.dist_class, self.estimate_batch)
        loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                Hv = hessian_vector_product(self.gradsH, self.params, v)
                tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.

        for i in range(maxIter):
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, iter=100, n_v=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            ############### Lanczos
            for i in range(iter):
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    w_prime = hessian_vector_product(self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    w_prime = hessian_vector_product(self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = torch.zeros(iter, iter).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            a_, b_ = torch.linalg.eig(T)

            eigen_list = a_
            weight_list = torch.pow(b_, 2)
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full