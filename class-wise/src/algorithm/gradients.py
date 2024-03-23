import torch
from torch.autograd import grad

def get_gradients(network, loss, create_graph=False):
    params = list(network.parameters())
    gradients = grad(loss, params, create_graph=create_graph)
    gradients = [x.view(-1) for x in gradients]
    return torch.cat(gradients)