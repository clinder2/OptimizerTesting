from CustomShampoo import *
from scipy.linalg import sqrtm

"""Squareroot-Cholesky-Inverse Shampoo"""
class SCIShampoo(CustomShampoo):
    def __init__(self, lr, W):
        super().__init__(lr, W)

    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                graft=self.state[p]['graft']
                grad=p.grad
                L=self.state[p]['L']
                R=self.state[p]['R']
                #print("DEVICE", L.device, grad.device)
                L=self.beta2*L+(1-self.beta2)*grad@grad.T #update left/right preconditioners
                R=self.beta2*R+(1-self.beta2)*grad.T@grad
                self.state[p]['L']=L #update state of preconditioners
                self.state[p]['R']=R

                L=self.sqrtm_newton_schulz(L)
                R=self.sqrtm_newton_schulz(R)

                Lp=torch.linalg.cholesky_ex(.001*torch.eye(L.shape[0],device=self.device)+L) #Cholesky decomp of L
                Rp=torch.linalg.cholesky_ex(.001*torch.eye(R.shape[0],device=self.device)+R) #Cholesky decomp of R
                if Lp.info==0 and Rp.info==0: #successful Cholesky decomp
                    # Lp=torch.linalg.inv_ex(Lp.L).inverse
                    # Rp=torch.linalg.inv_ex(Rp.L).inverse
                    Lp=Lp.L
                    Rp=Rp.L
                    Lp=torch.linalg.solve_triangular(Lp,torch.eye(Lp.shape[0],device=self.device),upper=False)
                    Rp=torch.linalg.solve_triangular(Rp,torch.eye(Rp.shape[0],device=self.device),upper=False)
                else: #standard Shampoo update, Cholesky failed
                    #Lp=self.mat_pow(L, 2)
                    #Rp=self.mat_pow(R, 2) #.T better
                    Lp=torch.eye(Lp.L.shape[0],device=self.device)
                    Rp=torch.eye(Rp.L.shape[0],device=self.device)
                    #print(f"failed on {self.iter}")
                    self.fails+=1
                update=Lp@grad@Rp.T
                
                graft.add_statistics(grad) #update grafting state
                graft_grad=graft.precondition_gradient(grad) #do grafting
                graft_n=torch.linalg.norm(graft_grad, ord='fro')
                shampoo_n=torch.linalg.norm(update, ord='fro')
                p.data-=g['lr']*(graft_n/(shampoo_n+1e-6))*update #param update with grafting
        self.iter+=1

    def sqrtm_newton_schulz(self, A, num_iters=10):
        """
        Computes the matrix square root of a positive semi-definite matrix using
        the Newton-Schulz iterative method in PyTorch.

        Args:
            A (torch.Tensor): A batch of square positive semi-definite matrices
                            of shape (*, n, n).
            num_iters (int): The number of iterations to run the algorithm.

        Returns:
            torch.Tensor: The matrix square root of A.
        """
        
        # Ensure input matrix has at least two dimensions
        if A.dim() < 2:
            raise ValueError(f"Input dimension equals {A.dim()}, expected at least 2")
        
        # Get the batch dimensions and matrix size
        batch_dims = A.shape[:-2]
        n = A.size(-1)
        
        # Check if the matrix is square
        if A.size(-1) != A.size(-2):
            raise ValueError("Input matrices must be square.")
        
        # Scale the matrices by their Frobenius norm for stable convergence
        # Frobenius norm is computed over the last two dimensions
        norm_A = torch.norm(A, p='fro', dim=(-2, -1), keepdim=True)
        
        # Handle the case where norm is close to zero to avoid division issues
        norm_A = torch.where(norm_A < 1e-8, 1e-8 * torch.ones_like(norm_A), norm_A)
        
        # Normalize the matrix and prepare the inverse estimate
        Y = A / norm_A
        I = torch.eye(n, n, dtype=A.dtype).expand(*batch_dims, n, n)
        Z = I.clone()
        
        # Perform Newton-Schulz iterations
        for _ in range(num_iters):
            T = 0.5 * (3.0 * I - Z@Y)
            Y = Y@T
            Z = T@Z
        
        # Apply the final scaling to get the actual square root
        return Y * torch.sqrt(norm_A)