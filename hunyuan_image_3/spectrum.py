"""Spectrum step-skipping for HunyuanImage-3.0.

Treats each image-token hidden-state channel as a function of sampling time, fits
Chebyshev polynomials T_0..T_M to the steps that really ran, and forecasts the
feature on skipped steps so only `final_layer` has to run.

Adapted from the Spectrum reference implementation:
  paper code   https://github.com/hanjq17/Spectrum  (src/utils/basis_utils.py)
  ComfyUI port https://github.com/judian17/ComfyUI-Spectrum  (forecaster.py, spectrum_state.py)
MIT License, Copyright (c) 2026 Jiaqi Han.
Changed here: the forward schedule is precomputed from the step list instead of
counting model calls, so it is sampler-agnostic, free of mutable state and
identical for both classifier-free-guidance passes.
"""

import logging
import math

import torch


def plan_schedule(num_steps, warmup_steps=3, window_size=2.0, flex_window=0.75):
    """Per-step run/skip plan: [(ran, skip_offset, window), ...], one entry per step.

    After `warmup_steps` every step runs, a step runs when
    (consecutive_skipped + 1) % floor(window) == 0, and `window` grows by
    `flex_window` after each real forward, so gaps widen towards the end.

    Two rules on top of the paper's: the first two steps always run (the
    forecaster needs two points before it can predict), and steps past
    `num_steps` are not planned here — `SpectrumState.should_run` runs them.
    """
    plan = []
    window = float(window_size)
    consecutive_skipped = 0
    actual = 0
    for step in range(max(int(num_steps), 0)):
        if step < warmup_steps or actual < 2:
            ran = True
            width = max(1, math.floor(window))
        else:
            width = max(1, math.floor(window))
            ran = (consecutive_skipped + 1) % width == 0
            if ran:
                window = round(window + flex_window, 3)
        if ran:
            actual += 1
            consecutive_skipped = 0
            plan.append((True, 0, max(1, math.floor(window))))
        else:
            consecutive_skipped += 1
            plan.append((False, consecutive_skipped, width))
    return plan


def plan_forward_steps(num_steps, warmup_steps=3, window_size=2.0, flex_window=0.75):
    """The same plan reduced to the run/skip flags."""
    return [ran for ran, _, _ in plan_schedule(num_steps, warmup_steps, window_size, flex_window)]


class ChebyshevForecaster:
    """Ridge regression of T_0..T_M in tau = 2t - 1 onto the stored features.

    Points are held in the dtype they arrive in (bf16 in the model); the fit and
    the prediction are float32.  At most `history` points are kept.
    """

    def __init__(self, M=4, lam=0.1, history=12):
        self.M = int(M)
        self.lam = float(lam)
        self.history = max(int(history), 2)
        self.times = []
        self.rows = []
        self.shape = None
        self.last_delta_norm = None
        self._coef = None

    @property
    def ready(self):
        return len(self.times) >= 2

    @property
    def nbytes(self):
        return sum(row.numel() * row.element_size() for row in self.rows)

    def update(self, t, hidden):
        row = hidden.detach().reshape(-1).clone()
        if self.shape is None:
            self.shape = tuple(hidden.shape)
        elif tuple(hidden.shape) != self.shape:
            raise ValueError(f"feature shape changed: {tuple(hidden.shape)} != {self.shape}")
        if self.rows:
            self.last_delta_norm = float((row.float() - self.rows[-1].float()).norm())
        self.times.append(float(t))
        self.rows.append(row)
        if len(self.rows) > self.history:
            self.times.pop(0)
            self.rows.pop(0)
        self._coef = None

    def _design(self, taus):
        """Chebyshev columns [T_0(tau), ..., T_M(tau)], shape (K, M + 1)."""
        taus = taus.reshape(-1, 1)
        columns = [torch.ones_like(taus)]
        if self.M >= 1:
            columns.append(taus)
        for _ in range(2, self.M + 1):
            columns.append(2 * taus * columns[-1] - columns[-2])
        return torch.cat(columns, dim=1)

    def _fit(self):
        if self._coef is not None:
            return self._coef
        if not self.ready:
            raise RuntimeError(f"need 2 points to fit, have {len(self.times)}")
        device = self.rows[0].device
        t = torch.tensor(self.times, dtype=torch.float32, device=device)
        X = self._design(2.0 * t - 1.0)
        Xt = X.transpose(0, 1)
        A = Xt @ X + self.lam * torch.eye(X.shape[1], dtype=torch.float32, device=device)
        # X^T H accumulated one stored row at a time: stacking the history in float32 first cost
        # K x F x 4 bytes (~0.8 GB at 1024^2, ~1.8 GB at 1536^2 with 12 points) on a card where every
        # spare GiB is weight residency. Same sum, peak of one float32 row.
        XtH = torch.zeros((X.shape[1], self.rows[0].numel()), dtype=torch.float32, device=device)
        for index, row in enumerate(self.rows):
            XtH.addr_(X[index], row.to(torch.float32))
        try:
            factor = torch.linalg.cholesky(A)
        except RuntimeError:
            jitter = 1e-6 * A.diag().mean()
            factor = torch.linalg.cholesky(A + jitter * torch.eye(A.shape[0], dtype=A.dtype, device=device))
        self._coef = torch.cholesky_solve(XtH, factor)
        return self._coef

    @torch.no_grad()
    def predict(self, t):
        coef = self._fit()
        tau = torch.tensor([2.0 * float(t) - 1.0], dtype=torch.float32, device=coef.device)
        out = (self._design(tau) @ coef).reshape(self.shape)
        return out.to(self.rows[-1].dtype)

    def clear(self):
        self.times = []
        self.rows = []
        self.shape = None
        self.last_delta_norm = None
        self._coef = None


class SpectrumForecaster:
    """Chebyshev prediction blended with a first-order Taylor extrapolation.

    h = (1 - w) * taylor + w * cheb, where `w` rises from the base value on the
    first skip of a window to `max_w` on the last one: longer forecasts lean on
    the polynomial, short ones on the local difference.
    """

    def __init__(self, M=4, lam=0.1, history=12, w=0.5, max_w=0.8):
        self.cheb = ChebyshevForecaster(M=M, lam=lam, history=history)
        self.w = float(w)
        self.max_w = float(max_w)

    @property
    def ready(self):
        return self.cheb.ready

    @property
    def nbytes(self):
        return self.cheb.nbytes

    @staticmethod
    def blend_weight(skip_offset, window, w, max_w):
        """Blend weight for a skip `skip_offset` steps into a window of `window`.

        A window of size n admits n - 1 skips, so the weight runs from `w` on the
        first to `max_w` on the last; a 2-step window has one skip and no ramp.
        """
        last = max(int(window) - 1, 1)
        offset = min(max(int(skip_offset), 1), last)
        span = max(last - 1, 1)
        return min(max_w, w + (max_w - w) * (offset - 1) / span)

    def update(self, t, hidden):
        self.cheb.update(t, hidden)

    def _taylor(self, t):
        rows, times = self.cheb.rows, self.cheb.times
        if len(times) < 2:
            return rows[-1].reshape(self.cheb.shape)
        # signed: on the sigma axis time *falls* from step to step, and clamping the step to a small
        # positive number made k enormous there (a linear feature forecast 2.6e5x off)
        step = times[-1] - times[-2]
        if abs(step) < 1e-8:
            return rows[-1].reshape(self.cheb.shape)
        k = (float(t) - times[-1]) / step
        out = rows[-1] + k * (rows[-1] - rows[-2])
        return out.reshape(self.cheb.shape).to(rows[-1].dtype)

    @torch.no_grad()
    def predict(self, t, skip_offset, window):
        weight = self.blend_weight(skip_offset, window, self.w, self.max_w)
        taylor = self._taylor(t)
        if self.cheb.ready:
            cheb = self.cheb.predict(t)
            return (1.0 - weight) * taylor + weight * cheb
        return taylor

    def clear(self):
        self.cheb.clear()


class SpectrumState:
    """Everything mutable for one sampling run: the plan, per-pass forecasters, stats.

    The skip decision is a lookup in a precomputed plan, so asking twice for the
    same step (one pass per classifier-free-guidance path) cannot advance it.

    Defaults follow the paper's configs/base.yaml (warmup_steps 5, window_size 2,
    flex_window = alpha 0.75); the port's warmup of 3 gives 12 passes at 50 steps
    rather than the paper's 14.
    """

    def __init__(self, num_steps=None, warmup_steps=5, window_size=2.0, flex_window=0.75,
                 w=0.5, max_w=0.8, M=4, lam=0.1, history=12, time_axis="step",
                 verbose=False, validate=False):
        if time_axis not in ("step", "sigma"):
            raise ValueError(f"time_axis must be 'step' or 'sigma', got {time_axis!r}")
        self.num_steps = None
        self.warmup_steps = int(warmup_steps)
        self.window_size = float(window_size)
        self.flex_window = float(flex_window)
        self.w = float(w)
        self.max_w = float(max_w)
        self.M = int(M)
        self.lam = float(lam)
        self.history = int(history)
        self.time_axis = time_axis
        self.verbose = bool(verbose)
        self.validate = bool(validate)
        self.plan = []
        self.forecasters = {}
        self.ran_steps = 0
        self.skipped_steps = 0
        self.validation_errors = []
        self._seen = set()
        if num_steps is not None:
            self.begin(num_steps)

    def begin(self, num_steps):
        """Fix the run length once the sampler's schedule is visible.

        A state built by the node starts without one: the step count belongs to the sampler, which
        the graph cannot know before it runs. Idempotent, so the step loop can call it every step.
        """
        num_steps = int(num_steps)
        if self.num_steps == num_steps:
            return
        if num_steps < 1:
            raise ValueError(f"a sampling run needs at least one step, got {num_steps}")
        self.clear()
        self.num_steps = num_steps
        self.plan = plan_schedule(num_steps, self.warmup_steps, self.window_size, self.flex_window)

    def should_run(self, step_index):
        """Whether this step runs the transformer. Past the plan, always run."""
        step_index = int(step_index)
        if 0 <= step_index < len(self.plan):
            return self.plan[step_index][0]
        return True

    def skip_position(self, step_index):
        """(skip_offset, window) for a step in the plan; (0, 1) when it runs."""
        step_index = int(step_index)
        if 0 <= step_index < len(self.plan):
            _, offset, window = self.plan[step_index]
            return offset, window
        return 0, 1

    def note_step(self, step_index, ran):
        """Count a step once, however many model calls it produces."""
        step_index = int(step_index)
        if step_index in self._seen:
            return
        self._seen.add(step_index)
        if ran:
            self.ran_steps += 1
        else:
            self.skipped_steps += 1

    def time_for(self, step_index, sigma=None, sigma_range=None):
        """Sampling time in [0, 1] for this step, on the configured axis."""
        if self.time_axis == "sigma" and sigma is not None and sigma_range is not None:
            low, high = sigma_range
            if high - low > 0:
                return float((float(sigma) - low) / (high - low))
        return int(step_index) / max((self.num_steps or 1) - 1, 1)

    def _forecaster(self, key):
        if key not in self.forecasters:
            self.forecasters[key] = SpectrumForecaster(
                M=self.M, lam=self.lam, history=self.history, w=self.w, max_w=self.max_w)
        return self.forecasters[key]

    def store(self, key, t, hidden):
        self._forecaster(key).update(t, hidden)

    @torch.no_grad()
    def predict(self, key, t, step_index):
        offset, window = self.skip_position(step_index)
        return self._forecaster(key).predict(t, offset, window)

    def log_validation(self, step_index, predicted, real):
        """Relative L2 error of a forecast against the real forward."""
        error = float((predicted.float() - real.float()).norm() / real.float().norm().clamp_min(1e-12))
        self.validation_errors.append((int(step_index), error))
        return error

    def peak_forecaster_bytes(self):
        return sum(forecaster.nbytes for forecaster in self.forecasters.values())

    def summary(self):
        speedup = (self.num_steps or 0) / max(self.ran_steps, 1)
        line = (f"[Spectrum] {self.num_steps} steps, {self.ran_steps} ran, "
                f"{self.skipped_steps} skipped, {speedup:.2f}x, "
                f"peak forecast memory {self.peak_forecaster_bytes() / 2**20:.0f} MiB")
        if self.validation_errors:
            worst = max(self.validation_errors, key=lambda pair: pair[1])
            line += f", worst forecast error {worst[1]:.4f} at step {worst[0]}"
        return line

    def report(self):
        """One line per run, as the plan asks. `verbose` only adds the per-step detail.

        Silent when the run never began, so a node that the sampler never reached does not
        report a step count it does not have.
        """
        if self.num_steps is not None:
            logging.info(self.summary())

    def clear(self):
        """Release the stored features, keeping the configuration."""
        for forecaster in self.forecasters.values():
            forecaster.clear()
        self.forecasters = {}
        self.validation_errors = []
        self._seen = set()
        self.ran_steps = 0
        self.skipped_steps = 0

    def reset(self):
        """Back to the state of a fresh run: a second run matches the first."""
        self.clear()