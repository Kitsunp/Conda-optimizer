import torch
import math
from torch.optim import Optimizer


class Conda(Optimizer):
    """
    Conda optimizer with Cautious Weight Decay (CWD) support.
    
    Based on official Conda implementation with CWD integration.
    Applies Conda projection to 2D parameters, standard Adam to 1D parameters.
    
    CWD applies weight decay only where the optimizer update direction
    aligns with the parameter sign (sign-selective regularization).
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1,
        correct_bias=True,
        update_proj_gap=500,
        scale=0.25,
        use_cwd=True,  # Cautious Weight Decay
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "use_cwd": use_cwd,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError("Conda does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                # State initialization
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                
                # Update first moment
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                
                # Conda Projection (only for 2D parameters)
                if grad.ndim == 2:
                    # Initialize projector state
                    if "projector_ortho" not in state:
                        state["projector_ortho"] = None
                        state["projector_last_svd_step"] = -1
                    
                    # Project grad and exp_avg
                    grad = self._project_with_cached_ortho(
                        grad, exp_avg, state, group["update_proj_gap"]
                    )
                    exp_avg = self._project_with_cached_ortho(
                        exp_avg, exp_avg, state, group["update_proj_gap"]
                    )
                
                # Update second moment (PROJECTED for 2D - critical for Conda stability)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                state["step"] += 1

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # Compute normalized gradient (in subspace for 2D)
                norm_grad = exp_avg / denom
                
                # Project back to parameter space (only for 2D parameters)
                # This is the ACTUAL update that Conda will apply to parameters
                if grad.ndim == 2 and "projector_ortho" in state:
                    norm_grad = self._project_back(norm_grad, state, group["scale"])

                # ============ CWD INTEGRATION ============
                # norm_grad is "u_t" in CWD Algorithm 1:
                # - For 2D: includes full Conda geometry (project→normalize→project back)
                # - For 1D: standard Adam update (exp_avg/denom)
                
                # Weight decay BEFORE parameter update (operates on p_t, not p_{t+1})
                if group["weight_decay"] > 0.0:
                    if group["use_cwd"]:
                        # Cautious Weight Decay: only decay where sign(u_t) == sign(p_t)
                        # Mask construction: 𝕀(u_t ⊙ x_t ≥ 0) from Algorithm 1
                        mask = (norm_grad * p) >= 0
                        # Apply WD only on masked coordinates
                        p.add_(p * mask, alpha=(-group["lr"] * group["weight_decay"]))
                    else:
                        # Standard decoupled weight decay (AdamW-style)
                        p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                # =========================================

                # Apply Conda update: x_{t+1} = x_t - η_t * u_t
                p.add_(norm_grad, alpha=-step_size)
                
        return loss
    
    def _project_with_cached_ortho(self, input_matrix, svd_basis_matrix, state, update_proj_gap):
        """Project matrix using cached orthogonal basis."""
        update_condition = (state["projector_ortho"] is None or 
                           state["step"] % update_proj_gap == 0)
        already_updated_this_step = state["step"] == state["projector_last_svd_step"]
        
        # Update orthogonal matrix if needed
        if update_condition and not already_updated_this_step:
            # Determine projection type based on shape (std behavior)
            if input_matrix.shape[0] >= input_matrix.shape[1]:
                state["projector_ortho"] = self._get_orthogonal_matrix(svd_basis_matrix, type='right')
                state["projector_type"] = 'right'
            else:
                state["projector_ortho"] = self._get_orthogonal_matrix(svd_basis_matrix, type='left')
                state["projector_type"] = 'left'
            
            state["projector_last_svd_step"] = state["step"]
        
        # Project
        device = input_matrix.device
        ortho = state["projector_ortho"]
        
        if state["projector_type"] == 'right':
            projected_matrix = torch.matmul(input_matrix, ortho.t().to(device))
        else:  # 'left'
            projected_matrix = torch.matmul(ortho.t().to(device), input_matrix)
        
        return projected_matrix
    
    def _project_back(self, projected_matrix, state, scale):
        """Project back to original space."""
        device = projected_matrix.device
        ortho = state["projector_ortho"]
        
        if state["projector_type"] == 'right':
            projected_back_matrix = torch.matmul(projected_matrix, ortho.to(device))
        else:  # 'left'
            projected_back_matrix = torch.matmul(ortho.to(device), projected_matrix)
        
        return projected_back_matrix * scale
    
    def _get_orthogonal_matrix(self, svd_basis_matrix, type):
        """Compute orthogonal matrix via SVD."""
        matrix = svd_basis_matrix.data
        orig_dtype = matrix.dtype
        orig_device = matrix.device

        # Perform SVD in float32 for numerical stability
        if orig_dtype != torch.float:
            matrix = matrix.float()

        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

        if type == 'right':
            result = Vh
        else:  # 'left'
            result = U
        
        # Convert back to original dtype
        if orig_dtype != torch.float:
            result = result.to(orig_device).type(orig_dtype)
        
        return result
