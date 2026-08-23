import torch
import time
from torch.optim import Optimizer
from Grafting import AdagradGraft

class StackedShampoo(Optimizer):
    def __init__(self, param_groups, grafting=False, p=4, chol=False, optimized=False, debug=False, numIters=20, **kwargs):
        super().__init__(param_groups, defaults={})
        self.device=param_groups[0]['params'][0].device
        #self.L=torch.eye(W[0].shape[0]) #left preconditioner
        #self.R=torch.eye(W[0].shape[0]) #right preconditioner
        self.p=p #matrix power (4 for -1/4, 2 for -1/2, etc.)
        self.chol=chol #cholesky or not
        self.opt=optimized #Anil approximation or not
        self.state={} #paramteter state dictionary
        for g in self.param_groups:
            g['initial_lr']=g['lr']
            shape = g['params'][0].shape
            num_params = len(g['params'])
            p=g['params'][0]
            self.state[p]={} #init each group's state
            self.state[p]['graft']=AdagradGraft(None, torch.stack([pm for pm in g['params']])) #init Adagrad grafting
            self.state[p]['L']=torch.eye(shape[0],device=self.device).unsqueeze(0).repeat(num_params,1,1) #group's left preconditioner
            self.state[p]['R']=torch.eye(shape[1],device=self.device).unsqueeze(0).repeat(num_params,1,1) #group's right preconditioner
            self.state[p]['total_time']=0
            self.state[p]['total_time_arr']=[]
            print("init shampoo: ", g['beta2'])
        self.debug=debug
        self.iter=0
        self.fails=0
        self.padL = torch.eye(shape[0],device=self.device).unsqueeze(0).repeat(num_params,1,1)
        self.padR = torch.eye(shape[1],device=self.device).unsqueeze(0).repeat(num_params,1,1)
        self.eps=0.001
        self.grafting=grafting
        self.numIters=numIters
        print('iters: ', self.numIters)

    def step(self):
        self.fails=0
        total=0
        for g in self.param_groups:
            state = self.state[g['params'][0]]
            stacked_params = torch.stack([p.data for p in g['params']])
            stacked_grads = torch.stack([p.grad for p in g['params']])
            graft = state['graft']
            L = state['L']
            R = state['R']
            L = torch.lerp(L, stacked_grads @ stacked_grads.mT, g['beta2'])  # update left/right stacked preconditioners
            R = torch.lerp(R, stacked_grads.mT @ stacked_grads, g['beta2'])
            state['L'] = L + self.eps*torch.eye(L.shape[1],device=self.device).unsqueeze(0).repeat(L.shape[0],1,1)
            state['R'] = R + self.eps*torch.eye(R.shape[1],device=self.device).unsqueeze(0).repeat(R.shape[0],1,1)
            total+=L.shape[0]
            if self.chol:

                Lp, infoL = torch.linalg.cholesky_ex(state['L']) #Cholesky decomp of L
                Rp, infoR = torch.linalg.cholesky_ex(state['R']) #Cholesky decomp of R
                failedCholL = infoL.nonzero(as_tuple=True)[0].tolist()
                failedCholR = infoR.nonzero(as_tuple=True)[0].tolist()

                #print("success: ", Lp.shape, len(failedCholL), len(failedCholR))
                # if len(failedCholL)>0:
                #    #print("L", Lp[failedCholL], stacked_grads[failedCholL])
                #    for m in failedCholL:
                #       print("L: ", torch.linalg.norm(Lp[m], ord='fro'))
                #       print("G: ", torch.linalg.norm(stacked_grads[m], ord='fro'))
                #       print("P: ", torch.linalg.norm(stacked_params[m], ord='fro'))
                #       print(f"max: {stacked_params[m].name}", torch.max(Lp[m]), torch.max(stacked_grads[m]), torch.max(stacked_params[m]))
                # if len(failedCholR)>0:
                #    #print("R", Rp[failedCholR], stacked_grads[failedCholR])
                #    for m in failedCholR:
                #       print(m)
                #       print("R: ", torch.linalg.norm(Rp[m], ord='fro'))
                #       print("G: ", torch.linalg.norm(stacked_grads[m], ord='fro'))
                #       print("P: ", torch.linalg.norm(stacked_params[m], ord='fro'))
                #       print("max: ", torch.max(Rp[m]), torch.max(stacked_grads[m]), torch.max(stacked_params[m]))

                if len(failedCholL)>0:
                  print('failedL')
                  Lp[failedCholL]=ComputePower(state['L'][failedCholL], self.p, iter_count=self.numIters)
                if len(failedCholR)>0:
                  print('failedR')
                  Rp[failedCholR]=ComputePower(state['R'][failedCholR], self.p, iter_count=self.numIters)
                successL=list(set(range(Lp.shape[0]))-set(failedCholL))
                successR=list(set(range(Rp.shape[0]))-set(failedCholR))
                if len(successL):
                  #Lp[successL]=inverse_sqrtm_newton_schulz(Lp[successL],20)
                  Lp[successL]=ComputePower(Lp[successL], self.p//2, iter_count=self.numIters)
                if len(successR):
                  #Rp[successR]=torch.linalg.solve_triangular(Rp[successR],torch.eye(Rp[0].shape[0],device=self.device).unsqueeze(0).repeat(Rp.shape[0],1,1),upper=True)

                  #Rp[successR]=inverse_sqrtm_newton_schulz(Rp[successR],20)
                  Rp[successR]=ComputePower(Rp[successR], self.p//2, iter_count=self.numIters)
                self.fails+=len(failedCholL)+len(failedCholR)

                # print("success: ", Lp.shape, len(failedCholL), len(failedCholR))
                # if len(failedCholL)>0:
                #    #print("L", Lp[failedCholL], stacked_grads[failedCholL])
                #    for m in failedCholL:
                #       print("L: ", torch.linalg.norm(Lp[m], ord='fro'))
                #       print("G: ", torch.linalg.norm(stacked_grads[m], ord='fro'))
                #       print("P: ", torch.linalg.norm(stacked_params[m], ord='fro'))
                #       print("max: ", torch.max(Lp[m]), torch.max(stacked_grads[m]), torch.max(stacked_params[m]))
                # if len(failedCholR)>0:
                #    #print("R", Rp[failedCholR], stacked_grads[failedCholR])
                #    for m in failedCholR:
                #       print(m)
                #       print("R: ", torch.linalg.norm(Rp[m], ord='fro'))
                #       print("G: ", torch.linalg.norm(stacked_grads[m], ord='fro'))
                #       print("P: ", torch.linalg.norm(stacked_params[m], ord='fro'))
                #       print("max: ", torch.max(Rp[m]), torch.max(stacked_grads[m]), torch.max(stacked_params[m]))

                update=Lp@stacked_grads@Rp#.mT
                #update=stacked_grads@Rp
            else: #just standard Shampoo

                Lp=ComputePower(state['L'], self.p, iter_count=self.numIters) #L^{-1/4}
                Rp=ComputePower(state['R'], self.p, iter_count=self.numIters) #R^{-1/4}

                # for m in range(Lp.shape[0]):
                #       print(m)
                #       print("R: ", torch.linalg.norm(Rp[m], ord='fro'))
                #       print("G: ", torch.linalg.norm(stacked_grads[m], ord='fro'))
                #       print("P: ", torch.linalg.norm(stacked_params[m], ord='fro'))

                #print("plain time: ", e-s)
                update=Lp@stacked_grads@Rp
            if self.grafting:
              graft.add_statistics(stacked_grads) #update grafting state
              graft_grad=graft.precondition_gradient(stacked_grads) #do grafting
              graft_n=torch.linalg.norm(graft_grad, ord='fro', dim=(-2,-1))
              shampoo_n=torch.linalg.norm(update, ord='fro', dim=(-2,-1))
              step = g['lr'] * (graft_n / (shampoo_n + 1e-6)).view(-1, 1, 1)
            else:
              step = g['lr']

            #state['total_time']+=e-s
            #state['total_time_arr'].append(e-s)

            updates = step * update
            for p, delta in zip(g['params'], updates.unbind(0)):
                p.data -= delta
                #print(torch.linalg.norm(p.data, ord='fro'))
            if self.debug and self.iter%10==0:
                print(f"PRECONDITIONERS at {self.iter}:")
                print("SHAPE: ", "Lp: ", Lp.shape, " Rp: ", Rp.shape)
                print("NORM (fro)", torch.linalg.norm(Lp,ord="fro",dim=(-2,-1)))
                if self.opt and Rp.inverse!=None:
                    print("R: ", Rp.inverse)
                else:
                    print("L: ", Lp.data)
                    print("R: ", Lp.data)
                print(f"UPDATE at {self.iter}:")
                print(update)

        #     if self.iter>0 and self.iter%10==0:
        #       print("ave_time: ", state['total_time']/self.iter)
        # print("chol fails: ", self.fails, total)
        self.iter+=1

    def zero_grad(self, set_to_none = True):
        super().zero_grad(set_to_none)

def inverse_sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int = 100):
    """
    Approximate the inverse square root of a matrix using the Newton-Schulz method.
    Adapted from https://discuss.pytorch.org/t/pytorch-square-root-of-a-positive-semi-definite-matrix/100138
    """
    if matrix.dim() < 2 or matrix.size(-1) != matrix.size(-2):
        raise ValueError("Input must be a square matrix or a batch of square matrices.")
    
    dim = matrix.size(-1)
    
    # Calculate the Frobenius norm of the matrix (batched)
    norm_of_matrix = torch.norm(matrix, p='fro', dim=(-2, -1), keepdim=True)
    
    # Normalize the matrix and initialize Y and Z
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
    Z = I.clone()

    #print(matrix.shape, Y.shape, I.shape, Z.shape)
    # Newton-Schulz iteration
    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.matmul(Y))
        Y = Y.matmul(T)
        Z = T.matmul(Z)
    
    # Rescale and return the approximate inverse square root
    result = Z.div(torch.sqrt(norm_of_matrix))

    return result

###
# Scalable Shampoo matrix power functions: 
# https://github.com/google-research/google-research/blob/master/scalable_shampoo/pytorch/matrix_functions.py 
###

@torch.no_grad()
def PowerIter(mat_g, error_tolerance=1e-6, num_iters=100):
  """Power iteration.

  Compute the maximum eigenvalue of mat, for scaling.
  v is a random vector with values in (-1, 1).

  Args:
    mat_g: the symmetric PSD matrix or a batch of such matrices.
    error_tolerance: Iterative exit condition.
    num_iters: Number of iterations.

  Returns:
    eigen values, eigen vectors, num_iters
  """
  if mat_g.ndim == 2:
    v = torch.rand(mat_g.shape[1], device=mat_g.device) * 2 - 1
    singular_val = torch.tensor(0.0, device=mat_g.device)
    error = torch.tensor(1.0, device=mat_g.device)
    iters = 0
    while error > error_tolerance and iters < num_iters:
      v = v / torch.norm(v)
      mat_v = torch.mv(mat_g, v)
      s_v = torch.dot(v, mat_v)
      error = torch.abs(s_v - singular_val)
      v = mat_v
      singular_val = s_v
      iters += 1
    return singular_val, v / torch.norm(v), iters

  batch_size, n, _ = mat_g.shape
  v = torch.rand(batch_size, n, device=mat_g.device) * 2 - 1
  singular_val = torch.zeros(batch_size, device=mat_g.device)
  error = torch.tensor(1.0, device=mat_g.device)
  iters = 0
  #print(v.shape, "v", singular_val.shape, "singular_val")
  while error > error_tolerance and iters < num_iters:
    v = v / torch.norm(v, dim=1, keepdim=True).clamp_min(1e-12)
    mat_v = (mat_g@v.unsqueeze(-1)).squeeze(-1)
    #print(mat_v.shape)
    s_v = (v*mat_v).sum(-1)
    #s_v = torch.sum(torch.bmm(v, mat_v), dim=1)
    ##print(s_v.shape)
    error = torch.max(torch.abs(s_v - singular_val))
    v = mat_v
    singular_val = s_v
    #print(singular_val.shape, "singular_val")
    iters += 1
  #print(singular_val.shape, "sing")
  return singular_val, v / torch.norm(v, dim=1, keepdim=True), iters

@torch.no_grad()
def MatPower(mat_m, p):
  """Computes mat_m^p, for p a positive integer.

  Args:
    mat_m: a square matrix or a batch of square matrices.
    p: a positive integer

  Returns:
    mat_m^p
  """
  if p in [1, 2, 4, 8, 16, 32]:
    p_done = 1
    res = mat_m
    while p_done < p:
      res = torch.matmul(res, res)
      p_done *= 2
    return res

  power = None
  while p > 0:
    if p % 2 == 1:
      power = torch.matmul(mat_m, power) if power is not None else mat_m
    p //= 2
    mat_m = torch.matmul(mat_m, mat_m)
  return power

@torch.no_grad()
def ComputePower(mat_g, p,
                 iter_count=100,
                 error_tolerance=1e-6,
                 ridge_epsilon=1e-6):
  """A method to compute G^{-1/p} using a coupled Newton iteration.

  See for example equation 3.2 on page 9 of:
  A Schur-Newton Method for the Matrix p-th Root and its Inverse
  by Chun-Hua Guo and Nicholas J. Higham
  SIAM Journal on Matrix Analysis and Applications,
  2006, Vol. 28, No. 3 : pp. 788-804
  https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

  Args:
    mat_g: A square positive semidefinite matrix
    p: a positive integer
    iter_count: Stop iterating after this many rounds.
    error_tolerance: Threshold for stopping iteration
    ridge_epsilon: We add this times I to G, to make is positive definite.
                   For scaling, we multiply it by the largest eigenvalue of G.
  Returns:
    (mat_g + rI)^{-1/p} (r = ridge_epsilon * max_eigenvalue of mat_g).
  """
  #print("ComputePower: ", p)
  if mat_g.ndim == 2:
    return torch.pow(mat_g + ridge_epsilon, -1 / p)

  batch_size, n, _ = mat_g.shape
  if batch_size==0:
    return None
  if n == 1:
    return torch.eye(1, device=mat_g.device).repeat(batch_size, 1, 1)

  #print(mat_g.shape)
  identity = torch.eye(n, device=mat_g.device).unsqueeze(0).repeat(batch_size, 1, 1)
  alpha = -1.0 / p
  max_ev, _, _ = PowerIter(mat_g, num_iters=50)
  #print(max_ev.shape, identity.shape, "maxmax")
  ridge_term = ridge_epsilon * max_ev.view(batch_size,1,1)
  #print(ridge_term.shape)
  mat_g = mat_g + ridge_term*identity
  z = (1 + p) / (2 * torch.norm(mat_g, dim=(-2, -1), keepdim=True))
  #print(z.shape, "z")
  # The best value for z is
  # (1 + p) * (c_max^{1/p} - c_min^{1/p}) /
  #            (c_max^{1+1/p} - c_min^{1+1/p})
  # where c_max and c_min are the largest and smallest singular values of
  # mat_g.
  # The above estimate assumes that c_max > c_min * 2^p
  # Can replace above line by the one below, but it is less accurate,
  # hence needs more iterations to converge.
  # z = (1 + p) / tf.trace(mat_g)
  # If we want the method to always converge, use z = 1 / norm(mat_g)
  # or z = 1 / tf.trace(mat_g), but these can result in many
  # extra iterations.

  mat_root = identity * torch.pow(z, 1.0/p)
  mat_m = mat_g * z
  error = torch.max(torch.abs(mat_m - identity))
  count = 0
  #print(mat_m.shape, "mat_m")
  while error > error_tolerance and count < iter_count:
    tmp_mat_m = (1 - alpha) * identity + alpha * mat_m
    new_mat_root = torch.bmm(mat_root, tmp_mat_m)
    mat_m = torch.bmm(MatPower(tmp_mat_m, p), mat_m)
    new_error = torch.max(torch.abs(mat_m - identity))
    if new_error > error * 1.2:
      break
    mat_root = new_mat_root
    error = new_error
    count += 1
  return mat_root
