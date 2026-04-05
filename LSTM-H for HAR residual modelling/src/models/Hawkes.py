"""
Exponential-kernel Hawkes process estimated by maximum likelihood.

References
----------
Hawkes1971
    Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting
    point processes. Biometrika, 58(1), 83-90.

Ogata1981
    Ogata, Y. (1981). On Lewis' simulation method for point processes.
    IEEE Transactions on Information Theory, 27(1), 23-31.

Notes
-----
The model specification follows Hawkes1971. The recursive update used to
evaluate the exponential-kernel intensity efficiently is the standard Ogata1981
recursion implemented here in the log-likelihood.
"""

# hawkes_exponential_mle_torch.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class HawkesFitResult:
    mu: float
    alpha: float
    beta: float
    branching_ratio: float
    neg_loglik: float
    converged: bool
    n_iter: int
    device: str


class ExponentialHawkesMLE(nn.Module):
    """
    Univariate Hawkes process with exponential kernel:

        lambda(t) = mu + alpha * sum_{t_j < t} exp(-beta * (t - t_j))

    Parameters are constrained to be positive via softplus transforms.
    The branching ratio alpha / beta is also monitored.
    """

    def __init__(
        self,
        init_mu: float = 0.1,
        init_alpha: float = 0.1,
        init_beta: float = 1.0,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float64,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.eps = eps
        self.dtype = dtype
        self.device_name = device
        self.softplus = nn.Softplus()

        # Unconstrained parameters; transformed with softplus in forward pass
        self.raw_mu = nn.Parameter(
            self._inverse_softplus(torch.tensor(init_mu, dtype=dtype, device=device))
        )
        self.raw_alpha = nn.Parameter(
            self._inverse_softplus(torch.tensor(init_alpha, dtype=dtype, device=device))
        )
        self.raw_beta = nn.Parameter(
            self._inverse_softplus(torch.tensor(init_beta, dtype=dtype, device=device))
        )

        self.to(device=device, dtype=dtype)

    @staticmethod
    def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
        # Stable enough for positive initial values
        return torch.log(torch.expm1(x))

    def positive_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.softplus(self.raw_mu) + self.eps
        alpha = self.softplus(self.raw_alpha) + self.eps
        beta = self.softplus(self.raw_beta) + self.eps
        return mu, alpha, beta

    def ogata_recursion(
        self,
        event_times: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute R_i recursively:
            R_1 = 0
            R_i = exp(-beta * (t_i - t_{i-1})) * (1 + R_{i-1}), i >= 2

        Returns:
            R: tensor of shape (N,)
        """
        n = event_times.numel()
        R = torch.zeros(n, dtype=self.dtype, device=event_times.device)

        if n <= 1:
            return R

        dt = event_times[1:] - event_times[:-1]
        if torch.any(dt <= 0):
            raise ValueError("event_times must be strictly increasing.")

        for i in range(1, n):
            R[i] = torch.exp(-beta * dt[i - 1]) * (1.0 + R[i - 1])

        return R

    def log_likelihood(
        self,
        event_times: torch.Tensor,
        T: Optional[torch.Tensor] = None,
        penalty_stationarity: float = 0.0,
    ) -> torch.Tensor:
        """
        Log-likelihood of the univariate exponential Hawkes process.

        Args:
            event_times: strictly increasing 1D tensor of event times in [0, T]
            T: observation horizon. If None, uses last event time
            penalty_stationarity: optional penalty to discourage alpha / beta >= 1

        Returns:
            scalar log-likelihood tensor
        """
        if event_times.ndim != 1:
            raise ValueError("event_times must be a 1D tensor.")

        if event_times.numel() == 0:
            raise ValueError("event_times must contain at least one event.")

        if T is None:
            T = event_times[-1]
        else:
            T = T.to(device=event_times.device, dtype=self.dtype)

        mu, alpha, beta = self.positive_parameters()

        R = self.ogata_recursion(event_times, beta)
        lam = mu + alpha * R

        if torch.any(lam <= 0):
            raise RuntimeError("Encountered non-positive intensity.")

        log_term = torch.sum(torch.log(lam))

        integral_term = mu * T + (alpha / beta) * torch.sum(
            1.0 - torch.exp(-beta * (T - event_times))
        )

        loglik = log_term - integral_term

        if penalty_stationarity > 0.0:
            branching_ratio = alpha / beta
            penalty = penalty_stationarity * torch.relu(branching_ratio - 0.999) ** 2
            loglik = loglik - penalty

        return loglik

    def negative_log_likelihood(
        self,
        event_times: torch.Tensor,
        T: Optional[torch.Tensor] = None,
        penalty_stationarity: float = 0.0,
    ) -> torch.Tensor:
        return -self.log_likelihood(
            event_times=event_times,
            T=T,
            penalty_stationarity=penalty_stationarity,
        )

    @torch.no_grad()
    def get_parameters(self) -> Dict[str, float]:
        mu, alpha, beta = self.positive_parameters()
        br = alpha / beta
        return {
            "mu": float(mu.item()),
            "alpha": float(alpha.item()),
            "beta": float(beta.item()),
            "branching_ratio": float(br.item()),
        }

    def fit(
        self,
        event_times: torch.Tensor,
        T: Optional[float] = None,
        lr: float = 1e-2,
        max_iter: int = 2000,
        optimizer_name: str = "adam",
        penalty_stationarity: float = 1e3,
        tol: float = 1e-9,
        verbose: bool = True,
    ) -> HawkesFitResult:
        """
        Fit parameters by maximizing the Hawkes log-likelihood.

        Args:
            event_times: 1D tensor of strictly increasing event times
            T: observation horizon; if None, uses last event time
            lr: learning rate
            max_iter: maximum optimization iterations
            optimizer_name: 'adam' or 'lbfgs'
            penalty_stationarity: penalty weight for alpha/beta >= 1
            tol: convergence tolerance on objective changes
            verbose: print progress

        Returns:
            HawkesFitResult
        """
        event_times = event_times.to(device=self.device_name, dtype=self.dtype)

        if T is None:
            T_tensor = event_times[-1]
        else:
            T_tensor = torch.tensor(T, device=self.device_name, dtype=self.dtype)

        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_name.lower() == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.parameters(),
                lr=lr,
                max_iter=20,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError("optimizer_name must be 'adam' or 'lbfgs'.")

        prev_loss = None
        converged = False
        n_iter_done = 0

        if optimizer_name.lower() == "lbfgs":
            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss = self.negative_log_likelihood(
                    event_times=event_times,
                    T=T_tensor,
                    penalty_stationarity=penalty_stationarity,
                )
                loss.backward()
                return loss

            for i in range(max_iter):
                loss = optimizer.step(closure)
                loss_value = float(loss.item())
                n_iter_done = i + 1

                if verbose and (i % 25 == 0 or i == max_iter - 1):
                    params = self.get_parameters()
                    print(
                        f"[{i+1:4d}] nll={loss_value:.8f} "
                        f"mu={params['mu']:.6f} "
                        f"alpha={params['alpha']:.6f} "
                        f"beta={params['beta']:.6f} "
                        f"alpha/beta={params['branching_ratio']:.6f}"
                    )

                if prev_loss is not None and abs(prev_loss - loss_value) < tol:
                    converged = True
                    break
                prev_loss = loss_value

        else:
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = self.negative_log_likelihood(
                    event_times=event_times,
                    T=T_tensor,
                    penalty_stationarity=penalty_stationarity,
                )
                loss.backward()
                optimizer.step()

                loss_value = float(loss.item())
                n_iter_done = i + 1

                if verbose and (i % 100 == 0 or i == max_iter - 1):
                    params = self.get_parameters()
                    print(
                        f"[{i+1:4d}] nll={loss_value:.8f} "
                        f"mu={params['mu']:.6f} "
                        f"alpha={params['alpha']:.6f} "
                        f"beta={params['beta']:.6f} "
                        f"alpha/beta={params['branching_ratio']:.6f}"
                    )

                if prev_loss is not None and abs(prev_loss - loss_value) < tol:
                    converged = True
                    break
                prev_loss = loss_value

        params = self.get_parameters()

        return HawkesFitResult(
            mu=params["mu"],
            alpha=params["alpha"],
            beta=params["beta"],
            branching_ratio=params["branching_ratio"],
            neg_loglik=float(
                self.negative_log_likelihood(
                    event_times=event_times,
                    T=T_tensor,
                    penalty_stationarity=penalty_stationarity,
                ).item()
            ),
            converged=converged,
            n_iter=n_iter_done,
            device=self.device_name,
        )


def prepare_event_times(
    event_times,
    dtype: torch.dtype = torch.float64,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Convert event times into a sorted 1D torch tensor.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.as_tensor(event_times, dtype=dtype, device=device).flatten()

    if x.numel() == 0:
        raise ValueError("event_times is empty.")

    x, _ = torch.sort(x)

    if torch.any(x[1:] <= x[:-1]):
        raise ValueError("event_times must be strictly increasing after sorting.")

    return x


if __name__ == "__main__":
    # Example usage:
    # Event times in a single observation window [0, T]
    sample_event_times = [0.12, 0.43, 0.77, 1.31, 1.9, 2.15, 2.9, 3.45, 4.02]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    events = prepare_event_times(sample_event_times, device=device)

    model = ExponentialHawkesMLE(
        init_mu=0.3,
        init_alpha=0.5,
        init_beta=1.2,
        device=device,
        dtype=torch.float64,
    )

    result = model.fit(
        event_times=events,
        T=5.0,
        lr=1e-2,
        max_iter=1500,
        optimizer_name="adam",   # can also use "lbfgs"
        penalty_stationarity=1e3,
        tol=1e-10,
        verbose=True,
    )

    print("\nFit result")
    print(result)