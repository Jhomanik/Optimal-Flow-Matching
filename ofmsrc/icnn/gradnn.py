import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.func import hessian, vmap


class GradNNGeneric(nn.Module):
    def __init__(self, batch_size=1024):
        super().__init__()        
        self.batch_size = batch_size
    
    def forward(self, input) -> torch.Tensor:
        raise NotImplementedError()
    
    def push(self, input, create_graph=True, retain_graph=True):
        '''
        Pushes input by using the gradient of the network. By default preserves the computational graph.
        # Apply to small batches.
        '''
        input.requires_grad_(True)
        if len(input) <= self.batch_size:
            pot_outputs = self.forward(input)
            output = autograd.grad(
                outputs=pot_outputs, inputs=input,
                create_graph=create_graph, retain_graph=retain_graph,
                only_inputs=True,
                grad_outputs=torch.ones_like(pot_outputs, requires_grad=False)
            )[0]
            return output
        else:
            output = torch.zeros_like(input, requires_grad=False)
            for j in range(0, input.size(0), self.batch_size):
                output[j: j + self.batch_size] = self.push(
                    input[j:j + self.batch_size],
                     create_graph=create_graph, retain_graph=retain_graph)
            return output
    
    def push_nograd(self, input):
        '''
        Pushes input by using the gradient of the network. Does not preserve the computational graph.
        Use for pushing large batches (the function uses minibatches).
        '''
        output = torch.zeros_like(input, requires_grad=False)
        for i in range(0, len(input), self.batch_size):
            input_batch = input[i:i+self.batch_size]
            output.data[i:i+self.batch_size] = self.push(
                input[i:i+self.batch_size],
                create_graph=False, retain_graph=False
            ).data
        return output
    
    def hessian_nograd(self, input, batch_size=None):
        '''
        Takes `input` of shape (bs, dim)
        Returns hessian` of shape (bs, 1, dim, dim)
        '''
        if batch_size is None:
            batch_size = self.batch_size
        hess = vmap(hessian(lambda x: self(x.unsqueeze(0)).squeeze(0)))(input)
        
        # gradient = self.push(input)
        # hessian = torch.zeros(
        #     *gradient.size(), self.dim,
        #     dtype=torch.float32,
        #     requires_grad=True,
        # )

        # hessian = torch.cat(
        #     [
        #         torch.autograd.grad(
        #             outputs=gradient[:, d], inputs=input,
        #             create_graph=True, retain_graph=True,
        #             only_inputs=True, grad_outputs=torch.ones(input.size()[0]).float().to(input)
        #         )[0][:, None, :]
        #         for d in range(self.dim)
        #     ],
        #     dim = 1
        # )
        return hess.detach()


class GradICNNGeneric(GradNNGeneric):

    def convexify(self):
        raise NotImplementedError()
