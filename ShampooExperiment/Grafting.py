import torch

class Graft:
  """Base class to perform grafting onto Shampoo. This class does no grafting.
  """

  def __init__(self, hps, unused_var):
    self.hps = hps

  def add_statistics(self, grad):
    pass

  def precondition_gradient(self, grad):
    return grad

  def update_momentum(self, update, unused_beta1):
    return update


class SGDGraft(Graft):
  """Graft using SGD+momentum.

  momentum maintains an exponentially weighted moving average of gradients.
  """

  def __init__(self, hps, var):
    super(SGDGraft, self).__init__(hps, var)
    self.momentum = torch.zeros_like(var.data)

  def update_momentum(self, update, beta1):
    self.momentum.mul_(beta1).add_(update)
    return self.momentum


class AdagradGraft(SGDGraft):
  """Graft using Adagrad.

  Essentially an implementation of Adagrad with momentum.
  """

  def __init__(self, hps, var):
    super(AdagradGraft, self).__init__(hps, var)
    self.statistics = torch.zeros_like(var.data)

  def add_statistics(self, grad):
    self.statistics.add_(grad * grad)

  def precondition_gradient(self, grad):
    return grad / (torch.sqrt(self.statistics) + 1e-6)