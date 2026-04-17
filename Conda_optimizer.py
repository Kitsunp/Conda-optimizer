import torch
import math
from torch.optim import Optimizer


from torch.optim import Optimizer
class Conda(Optimizer):
    """
    Conda (Column-Normalized Adam) optimizer with:
    - CWD (Cautious Weight Decay): Standard sign-selective decay for multipliers
    - CHD (Cautious Huber Decay): CWD + Huber clipping + Correction for matrices/1D params
    - GPA (Generalized Primal Averaging): Optional smoothed iterate sequences
    - Conda Projection: Optional SVD-based subspace projection for 2D parameters

    Conda applies SVD-based projection to 2D parameters (weight matrices) for improved
    training stability and convergence in large language models. For 1D parameters
    (biases, layer norms), it behaves as standard Adam.

    CWD (base) is a sign-selective weight decay that only applies regularization
    when the optimizer update direction aligns with the parameter sign, preserving
    the original loss landscape while inducing better generalization.

    CHD extends CWD with:
    1. Proximal Huber operator: restricts regularization force for large weights acting as a strictly non-expansive proximal shrink rather than an explicit Euler step.
    2. Correction: adapts λ_t ∝ γ_t during LR decay (temporal)

    This combination provides:
    - Bounded and strictly non-expansive regularization (Proximal Huber)
    - Temporal stability during decay (Correction)
    - Directional preservation (Cautious)

    SPECTRA (Post-Spectral Clipping): Optional spectral wrapper per 2D parameter
    Based on "Enhancing LLM Training via Spectral Clipping" (arXiv:2603.14315).
    Applies mathematically exact Soft Spectral Clipping (SSC) post-optimizer update
    using a hardware-efficient bilateral Newton-Schulz iteration in bfloat16.
    Ensures the maximum singular value of the finalized update matrix does not strictly exceed
    alpha_sp * c_t, where c_t = c * (initial_lr / curr_lr) during warmup, remaining completely agnostic
    to Conda's internal geometry.
    When use_spectra=True, the update U_t is replaced identically by:
    U_t_final = max(sqrt(m/n), 1) * SSC_c_t(U_t_Conda)

    GPA (Generalized Primal Averaging) - Optional per group:
    GPA (arXiv:2512.17131) replaces the two-loop structure of DiLoCo with smooth
    primal averaging. It maintains two sequences:

    - Base iterate (z): Accumulates raw optimizer updates and weight decay
    - Smoothed iterate (x): Exponential moving average of z for stable evaluation

    The model parameters (p) are set to an interpolation between x and z:
      p = μ_y·x + (1-μ_y)·z

    This decouples:
    - Evaluation point (p): Where gradients are computed (interpolated)
    - Update accumulation (z): Where momentum and decay operate (base)
    - Smoothing (x): Long-term averaged weights (stable)

    GPA provides:
    - Faster convergence through iterate smoothing (Theory 3.1 in paper)
    - Reduced sensitivity to hyperparameters μ_x, μ_y
    - Better generalization via implicit regularization of smoothing

    When use_gpa=False: Reverts to standard Conda/Adam behavior where updates and
    decay operate directly on model parameters (p).

    Conda Projection - Optional per group via use_conda_proj:
    When use_conda_proj=False for a parameter group, the SVD-based subspace projection
    is completely skipped. The update rule becomes pure Adam (diagonal preconditioner,
    no subspace compression, no scale factor). Combined with use_gpa=False and
    apply_wd=False, this recovers standard AdamW for that group.

    This flag exists to accommodate architectures whose core parameters are not
    well-described by the low-rank gradient hypothesis that Conda assumes for 2D
    matrices. For example, Leviathan's separable surface (codebooks, spline_coeff)
    and JTok-M surfaces are parameterized by tensors of order ≥ 3 that already
    fall outside Conda projection. Their 2D interface matrices (W_out, W_res,
    seed_proj) share a coadaptive gradient chain with those tensors, making
    independent subspace projection of the interfaces potentially disruptive to
    the joint optimization geometry. Setting use_conda_proj=False on such groups
    allows Adam-like optimization with full per-coordinate adaptivity while keeping
    the group within a single optimizer instance.

    Hybrid regime summary:
    - Backbone 2D matrices:     use_conda_proj=True,  use_gpa=True   → full Conda + GPA
    - Leviathan / JTok-M:       use_conda_proj=False, use_gpa=False  → pure Adam (≈ AdamW)
    - Learnable multipliers:    use_conda_proj=False, use_gpa=True   → Adam + GPA, CWD only
    - Embeddings / bias / 1D:   use_conda_proj=False, apply_wd=False → Adam, no decay

    When use_gpa=False: Reverts to standard Conda behavior where updates and
    decay operate directly on model parameters (p), matching base Conda paper.

    References:
        - GPA paper: "Smoothing DiLoCo with Primal Averaging for Faster Training of LLMs"
          (arXiv:2512.17131, 2025)
        - Conda paper: "Column-Normalized Adam for Training Large Language Models Faster"
          (arXiv:2509.24218)
        - CWD paper: "Cautious Weight Decay" (arXiv:2510.12402)
        - Correction paper: "Correction of Decoupled Weight Decay" (arXiv:2512.08217)
        - Learnable Multipliers paper: "Learnable Multipliers: Freeing the Scale of
          Language Model Matrix Layers" (arXiv:2601.04890)
        - AdamHD paper: "AdamHD: Decoupled Huber Decay Regularization for Language
          Model Pre-Training" (arXiv:2511.14721)
        - Leviathan paper: "A Separable Architecture for Continuous Token Representation
          in Language Models" (arXiv:2601.22040)
        - JTok paper: "JTok: On Token Embedding as Another Axis of Scaling Law via
          Joint Token Self-Modulation" (arXiv:2602.00800)
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
        use_cwd=True,
        use_gpa=False,
        mu_x=0.9967,
        mu_y=0.9,
        huber_c=1.5,
        huber_beta=0.99,
        use_conda_proj=True,  # Global default — can be overridden per group
        use_spectra=False,    # Soft Spectral Clipping
        spectra_c=1.0,        # Spectral clipping threshold
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        if huber_c <= 0.0:
            raise ValueError(f"Invalid huber_c: {huber_c} - should be > 0.0")
        if use_gpa:
            if not 0.0 <= mu_x < 1.0:
                raise ValueError(f"Invalid mu_x: {mu_x} - should be in [0.0, 1.0)")
            if not 0.0 <= mu_y < 1.0:
                raise ValueError(f"Invalid mu_y: {mu_y} - should be in [0.0, 1.0)")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "use_cwd": use_cwd,
            "use_gpa": use_gpa,
            "mu_x": mu_x,
            "mu_y": mu_y,
            # Conda projection flag — set False per group to recover pure Adam
            # for parameters whose gradient geometry is not well-modelled by
            # the low-rank 2D subspace hypothesis (e.g. Leviathan / JTok-M).
            "use_conda_proj": use_conda_proj,
            "use_spectra": use_spectra,
            "spectra_c": spectra_c,
            # Huber Decay parameters
            "use_huber": False,
            "huber_c": huber_c,
            "huber_beta": huber_beta,
            "huber_use_ema": False,
            # Correction of Decoupled Weight Decay parameters
            "use_correction": False,
            "correction_kappa": None,
            # Learnable Multipliers support
            "apply_wd": True,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            group["max_lr_seen"] = 0.0
            group["correction_active"] = False
            group["correction_kappa"] = None
            if group.get("huber_use_ema", False):
                group["huber_delta_ema"] = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # ============ AUTOMATIC DECAY DETECTION ============
            if group.get("use_correction", True):
                current_lr = group["lr"]
                if current_lr > group["max_lr_seen"]:
                    group["max_lr_seen"] = current_lr
                if current_lr < group["max_lr_seen"] and not group["correction_active"]:
                    if group["weight_decay"] > 0.0:
                        group["correction_kappa"] = group["weight_decay"] / group["max_lr_seen"]
                        group["correction_active"] = True
                        first_param = group["params"][0]
                        if first_param in self.state and "step" in self.state[first_param]:
                            step = self.state[first_param]["step"]
                            print(f"📉 WD Correction activated at step {step}")
                            print(f"   max_lr_seen = {group['max_lr_seen']:.6f}")
                            print(f"   κ = {group['correction_kappa']:.6f}")
                            print(f"   Phase: Warmup/Plateau (λ=const) → Decay (λ∝γ)")
            # ===================================================

            use_gpa          = group["use_gpa"]
            use_conda_proj   = group.get("use_conda_proj", True)
            mu_x             = group["mu_x"] if use_gpa else None
            mu_y             = group["mu_y"] if use_gpa else None

            apply_wd  = group.get("apply_wd", True) and group["weight_decay"] > 0.0
            lambda_t  = self._compute_lambda_t(group) if apply_wd else 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "Conda does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if "exp_avg" not in state:
                    state["exp_avg"]    = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    if use_gpa:
                        state["z_buffer"] = p.clone().detach()
                        state["x_buffer"] = p.clone().detach()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                if use_gpa:
                    z = state["z_buffer"]
                    x = state["x_buffer"]
                else:
                    z = p

                beta1, beta2 = group["betas"]

                # Update first moment (always)
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))

                # ── Conda Projection ──────────────────────────────────────────────
                # Only for 2D parameters AND when use_conda_proj=True for this group.
                # When use_conda_proj=False, the update degenerates to pure Adam:
                # diagonal preconditioner, full-rank second moment, no scale factor.
                # Combined with use_gpa=False and apply_wd=False this is AdamW.
                if grad.ndim == 2 and use_conda_proj:
                    if "projector_ortho" not in state:
                        state["projector_ortho"]       = None
                        state["projector_last_svd_step"] = -1
                    grad    = self._project_with_cached_ortho(
                        grad, exp_avg, state, group["update_proj_gap"]
                    )
                    exp_avg = self._project_with_cached_ortho(
                        exp_avg, exp_avg, state, group["update_proj_gap"]
                    )
                # ─────────────────────────────────────────────────────────────────

                # Second moment updated in projected space (if projected) or full space
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                state["step"] += 1

                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    exp_avg_corrected    = exp_avg    / bias_correction1
                    exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                else:
                    exp_avg_corrected    = exp_avg
                    exp_avg_sq_corrected = exp_avg_sq

                denom     = exp_avg_sq_corrected.sqrt().add_(group["eps"])
                norm_grad = exp_avg_corrected / denom

                # Project back only if projection was applied to this group
                if grad.ndim == 2 and use_conda_proj and "projector_ortho" in state:
                    norm_grad = self._project_back(norm_grad, state, group["scale"])

                # ==================================================
                # SPECTRA: POST-SPECTRAL CLIPPING
                # Enhancing LLM Training via Spectral Clipping (arXiv:2603.14315)
                # Actúa de forma optimizer-agnostic sobre la dirección final de Conda
                # ANTES de calcular CWD y el Gradient Step final.
                # ==================================================
                if group.get("use_spectra", False) and grad.ndim == 2:
                    m, n = grad.shape
                    
                    # 1. Factor de escala espectral explícito de SPECTRA
                    alpha_sp = max(math.sqrt(m / n), 1.0)
                    
                    # 2. Agenda dinámica de c_t (Relaja el clip durante el warmup)
                    curr_lr = group["lr"]
                    # LambdaLR guarda la tasa de aprendizaje peak/base en "initial_lr"
                    base_lr = group.get("initial_lr", group["lr"])
                    
                    if curr_lr > 0 and curr_lr < base_lr and not group.get("correction_active", False):
                        # Fase de WARMUP: Relajamos el umbral para no asfixiar el modelo
                        c_t = group["spectra_c"] * (base_lr / curr_lr)
                    else:
                        # Fase de PLATEAU o DECAIMIENTO: Usamos el umbral estándar
                        c_t = group["spectra_c"]
                    
                    # 3. Aplicamos SSC_c_t suave (Newton-Schulz via bfloat16)
                    clipped_u = self._apply_spectral_clipping(norm_grad, c_t)
                    
                    # 4. Actualizamos la norma con ambos tensores
                    norm_grad = clipped_u * alpha_sp

                # ============ WEIGHT DECAY: CWD or CHD ============
                if apply_wd:
                    if group["use_cwd"]:
                        mask = (norm_grad * z) >= 0
                    else:
                        mask = torch.ones_like(z, dtype=torch.bool)

                    tau_t = group["lr"] * lambda_t

                    if group.get("use_huber", False):
                        # 1. Compute delta on z_t before update
                        delta_t = self._compute_huber_threshold(z, group, state)
                        
                        # 2. Base optimizer update FIRST (needed for Proximal operation)
                        z.add_(norm_grad, alpha=-group["lr"])
                        
                        # 3. Proximal AdamHD (applied selectively via mask)
                        limit = (1.0 + tau_t) * delta_t
                        
                        z_quad = z / (1.0 + tau_t)
                        z_lin = z - tau_t * delta_t * torch.sign(z)
                        
                        cond_quad = z.abs() <= limit
                        z_prox = torch.where(cond_quad, z_quad, z_lin)
                        
                        # Apply proximal form only where CWD mask allows
                        z.copy_(torch.where(mask, z_prox, z))
                    else:
                        # Explicit Euler Weight Decay
                        z.add_(z * mask, alpha=-tau_t)
                        z.add_(norm_grad, alpha=-group["lr"])
                else:
                    z.add_(norm_grad, alpha=-group["lr"])
                # ==================================================

                # ============ GPA SMOOTHING AND INTERPOLATION ============
                if use_gpa:
                    x.mul_(mu_x).add_(z, alpha=1.0 - mu_x)
                    p.copy_(x).mul_(mu_y).add_(z, alpha=1.0 - mu_y)
                # =========================================================

        return loss

    def _compute_lambda_t(self, group):
        """
        Compute weight decay coefficient.

        For CHD groups (use_correction=True):
          Phase 1 (Warmup/Plateau): λ_t = λ_0 (constant, full strength)
          Phase 2 (Decay): λ_t = κ·γ_t where κ = λ_0/γ_max (proportional to LR)

        For CWD groups (use_correction=False):
          Always: λ_t = λ_0 (constant)

        Returns:
            float: Weight decay coefficient for current step
        """
        if not group.get("use_correction", True):
            return group["weight_decay"]

        if group.get("correction_active", False):
            gamma_t = group["lr"]
            kappa   = group.get("correction_kappa", None)
            if kappa is not None:
                return kappa * gamma_t
            return group["weight_decay"]

        return group["weight_decay"]

    def _compute_huber_threshold(self, z, group, state):
        """
        Compute adaptive Huber threshold δ_t.

        From AdamHD paper, two strategies:
        1. Mean-magnitude (instant): δ_t = c·mean(|z_t|)
        2. EMA-based: δ_t = c·μ_t where μ_t = β·μ_{t-1} + (1-β)·mean(|z_t|)

        Args:
            z: Current working parameter (z_buffer if GPA, p if not)
            group: Parameter group dict
            state: Optimizer state for this parameter

        Returns:
            float: Adaptive threshold δ_t (scalar)
        """
        mean_abs = z.abs().mean().item()

        if group.get("huber_use_ema", False):
            beta = group["huber_beta"]
            if "huber_delta_ema" not in state:
                state["huber_delta_ema"] = mean_abs
            state["huber_delta_ema"] = (
                beta * state["huber_delta_ema"] + (1 - beta) * mean_abs
            )
            return group["huber_c"] * state["huber_delta_ema"]

        return group["huber_c"] * mean_abs

    def _project_with_cached_ortho(self, input_matrix, svd_basis_matrix, state, update_proj_gap):
        """Project matrix using cached orthogonal basis."""
        update_condition       = (state["projector_ortho"] is None or
                                  state["step"] % update_proj_gap == 0)
        already_updated        = state["step"] == state["projector_last_svd_step"]

        if update_condition and not already_updated:
            if input_matrix.shape[0] <= input_matrix.shape[1]:
                state["projector_ortho"] = self._get_orthogonal_matrix(
                    svd_basis_matrix, type='left'
                )
                state["projector_type"] = 'left'
            else:
                state["projector_ortho"] = self._get_orthogonal_matrix(
                    svd_basis_matrix, type='right'
                )
                state["projector_type"] = 'right'
            state["projector_last_svd_step"] = state["step"]

        device = input_matrix.device
        ortho  = state["projector_ortho"]
        if state["projector_type"] == 'right':
            return torch.matmul(input_matrix, ortho.t().to(device))
        else:
            return torch.matmul(ortho.t().to(device), input_matrix)

    def _project_back(self, projected_matrix, state, scale):
        """Project back to original space."""
        device = projected_matrix.device
        ortho  = state["projector_ortho"]
        if state["projector_type"] == 'right':
            res = torch.matmul(projected_matrix, ortho.to(device))
        else:
            res = torch.matmul(ortho.to(device), projected_matrix)
        return res * scale

    def _get_orthogonal_matrix(self, svd_basis_matrix, type):
        """Compute orthogonal matrix via SVD."""
        matrix     = svd_basis_matrix.data
        orig_dtype = matrix.dtype
        if orig_dtype != torch.float:
            matrix = matrix.float()
        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)
        result   = Vh if type == 'right' else U
        return result.to(svd_basis_matrix.device).type(orig_dtype)

    @torch.no_grad()
    def _apply_spectral_clipping(self, U, c, steps=10):
        """Soft Spectral Clipping (SSC) via Newton-Schulz iteration."""
        m, n = U.shape
        # Operaciones masivas en bfloat16 para máxima velocidad en Tensor Cores
        U_bf = U.bfloat16()
        device = U.device
        
        # 1. Construir la matriz simétrica A = I + (UU^T)/c^2 
        # (Siempre operando sobre la dimensión menor)
        if m <= n:
            A = torch.eye(m, device=device, dtype=torch.bfloat16) + (U_bf @ U_bf.mT) / (c * c)
        else:
            A = torch.eye(n, device=device, dtype=torch.bfloat16) + (U_bf.mT @ U_bf) / (c * c)
            
        # 2. Cota segura y ajustada mediante Norma Infinito (Gershgorin puro)
        alpha_ns = A.abs().sum(dim=-1).max().item()
        if alpha_ns <= 1e-6:
            return U
            
        # 3. Inicializar variables Newton-Schulz
        Y = A / alpha_ns
        Z = torch.eye(Y.size(0), device=device, dtype=torch.bfloat16)
        I = Z.clone()
        
        # 4. Iteración multiplicativa Newton-Schulz (T_k = 0.5 * (3I - Z_k Y_k))
        for _ in range(steps):
            T = 0.5 * (3.0 * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z
            
        # 5. Aproximar A^{-1/2}
        A_inv_sqrt = Z / math.sqrt(alpha_ns)
        
        # 6. Reconstruir el update ya "clippeado"
        if m <= n:
            U_clipped = A_inv_sqrt @ U_bf
        else:
            U_clipped = U_bf @ A_inv_sqrt
            
        return U_clipped.type_as(U)
        # Convert back to original dtype
        if orig_dtype != torch.float:
            result = result.to(orig_device).type(orig_dtype)
        
        return result
