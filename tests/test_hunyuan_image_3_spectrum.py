"""M1 tests for the Spectrum forecaster and its forward schedule.

CPU only: `hunyuan_image_3/spectrum.py` imports nothing from ComfyUI, so the schedule and the
forecaster are testable without a model, a tokenizer or a GPU.

The forward-step lists are written out literally rather than recomputed, because the schedule is
the thing being pinned. With `warmup_steps=5` they must reproduce the counts the paper publishes
for 50 steps (14 passes at `flex_window=0.75`, 10 at 3.0); that agreement is what fixes the rule.
Note the paper's own default is `warmup_steps: 5` (configs/base.yaml), not the port's 3 — at 3 the
same settings give 12 passes, i.e. a schedule more aggressive than the published one.
"""

import pytest
import torch

from hunyuan_image_3.spectrum import (
    ChebyshevForecaster,
    SpectrumForecaster,
    SpectrumState,
    plan_forward_steps,
    plan_schedule,
)

FIFTY_WARMUP5_FLEX075 = [0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 25, 31, 38, 46]
FIFTY_WARMUP5_FLEX30 = [0, 1, 2, 3, 4, 6, 11, 19, 30, 44]
EIGHT_WARMUP3_FLEX075 = [0, 1, 2, 4, 6]


def forward_steps(num_steps, warmup, window, flex):
    return [step for step, ran in enumerate(plan_forward_steps(num_steps, warmup, window, flex)) if ran]


def play(state, steps=8):
    """One simulated sampling run, two guidance passes per step. Returns the forecasts made."""
    forecasts = []
    for step in range(steps):
        ran = state.should_run(step)
        t = state.time_for(step)
        for key in (0, 1):
            feature = torch.full((4, 4), step / steps, dtype=torch.bfloat16)
            if ran:
                state.store(key, t, feature)
            else:
                forecasts.append((step, key, state.predict(key, t, step).clone()))
        state.note_step(step, ran)
    return forecasts


# ---- schedule ----

def test_schedule_reproduces_the_paper_counts():
    assert forward_steps(50, 5, 2.0, 0.75) == FIFTY_WARMUP5_FLEX075
    assert forward_steps(50, 5, 2.0, 3.0) == FIFTY_WARMUP5_FLEX30
    assert len(FIFTY_WARMUP5_FLEX075) == 14
    assert len(FIFTY_WARMUP5_FLEX30) == 10


def test_schedule_eight_steps():
    assert forward_steps(8, 3, 2.0, 0.75) == EIGHT_WARMUP3_FLEX075


def test_schedule_window_one_never_skips():
    """The M2 bit-identity gate: (n + 1) % 1 is always 0, so every step runs."""
    assert all(plan_forward_steps(50, 3, 1.0, 0.0))


def test_schedule_always_runs_the_first_two_steps():
    """Skipping before two points are stored would leave nothing to fit."""
    for warmup in (0, 1):
        assert plan_forward_steps(50, warmup, 8.0, 0.0)[:2] == [True, True]


def test_skip_offsets_stay_inside_their_window():
    for _, offset, window in plan_schedule(50, 5, 2.0, 0.75):
        assert 0 <= offset < window


# ---- the skip decision ----

def test_skip_decision_is_pure_and_pass_independent():
    state = SpectrumState(50, warmup_steps=5)
    first = [state.should_run(step) for step in range(50)]
    backward = [state.should_run(step) for step in reversed(range(50))][::-1]
    assert backward == first
    for _ in range(3):
        assert [state.should_run(step) for step in range(50)] == first
    assert first == [ran for ran, _, _ in state.plan]


def test_steps_past_the_plan_run():
    """A sampler calling the model more often than there are steps must not skip blindly."""
    state = SpectrumState(8, warmup_steps=3)
    assert state.should_run(99) is True
    assert state.skip_position(99) == (0, 1)


def test_steps_are_counted_once_however_many_calls():
    state = SpectrumState(8, warmup_steps=3)
    for step in range(8):
        for _pass in (0, 1):
            state.note_step(step, state.should_run(step))
    assert (state.ran_steps, state.skipped_steps) == (5, 3)


def test_summary_reports_the_measured_forward_count():
    state = SpectrumState(50, warmup_steps=5)
    for step in range(50):
        state.note_step(step, state.should_run(step))
    assert state.ran_steps == 14
    assert state.skipped_steps == 36
    assert "3.57x" in state.summary()


# ---- the forecaster ----

def test_chebyshev_recovers_a_polynomial_of_degree_m():
    """With lam ~ 0 and >= M + 1 points, a degree-M signal must come back exactly."""
    torch.manual_seed(0)
    degree = 4
    coefficients = torch.randn(degree + 1)

    def polynomial(t):
        return sum(coefficients[m] * t ** m for m in range(degree + 1))

    forecaster = ChebyshevForecaster(M=degree, lam=1e-8, history=12)
    for t in torch.linspace(0.05, 0.95, 9):
        forecaster.update(float(t), polynomial(t).expand(6, 3).clone())

    for t in (0.02, 0.31, 0.5, 0.7331, 0.999):
        want = polynomial(torch.tensor(t)).expand(6, 3)
        got = forecaster.predict(t).float()
        assert torch.allclose(got, want, atol=1e-4), (t, float((got - want).abs().max()))


def test_chebyshev_does_not_absorb_a_higher_degree():
    """Negative control: a degree-(M+1) signal must not fit, or the basis would not be degree M.

    Measured as the worst residual over a grid: the fit's residual is a degree-(M+1) polynomial
    with an exact zero at t=0.5 on a symmetric grid, so a single point can read as a false pass.
    """
    forecaster = ChebyshevForecaster(M=4, lam=1e-8, history=12)
    for t in torch.linspace(0.05, 0.95, 9):
        forecaster.update(float(t), (t ** 5).reshape(1, 1).clone())
    worst = max(
        abs(forecaster.predict(float(t)).float().item() - float(t) ** 5)
        for t in torch.linspace(0.0, 1.0, 21)
    )
    assert worst > 1e-3, worst


def test_prediction_keeps_the_stored_dtype():
    """model.py feeds bf16 and must not have to cast the result back."""
    forecaster = ChebyshevForecaster(M=4, lam=0.1, history=12)
    feature = torch.randn(4, 8, dtype=torch.bfloat16)
    for t in (0.0, 0.3, 0.6):
        forecaster.update(t, feature)
    assert forecaster.predict(0.9).dtype == torch.bfloat16


def test_history_caps_the_stored_points():
    forecaster = ChebyshevForecaster(M=4, lam=0.1, history=6)
    for step in range(20):
        forecaster.update(step / 19, torch.full((3, 3), float(step)))
    assert len(forecaster.rows) == 6
    assert forecaster.times[0] == pytest.approx(14 / 19)
    assert forecaster.times[-1] == pytest.approx(1.0)
    assert forecaster.nbytes == 6 * 3 * 3 * 4


def test_blend_weight_grows_across_a_window_and_caps_at_max_w():
    weights = [SpectrumForecaster.blend_weight(offset, 5, 0.5, 0.8) for offset in (1, 2, 3, 4, 5)]
    assert weights[0] == pytest.approx(0.5)
    assert weights[-1] == pytest.approx(0.8)
    assert weights == sorted(weights)
    assert SpectrumForecaster.blend_weight(9, 5, 0.5, 0.8) == pytest.approx(0.8)
    # a 2-step window admits one skip, so there is no ramp to make
    assert SpectrumForecaster.blend_weight(1, 2, 0.5, 0.8) == pytest.approx(0.5)


def test_predict_before_two_points_falls_back_to_taylor():
    forecaster = SpectrumForecaster(M=4, lam=0.1, history=12, w=0.5, max_w=0.8)
    forecaster.update(0.0, torch.ones(3, 4, dtype=torch.bfloat16))
    out = forecaster.predict(0.5, 1, 2)
    assert out.shape == (3, 4)
    assert out.dtype == torch.bfloat16


def test_a_changed_feature_shape_is_rejected():
    forecaster = SpectrumForecaster(M=4, lam=0.1, history=12, w=0.5, max_w=0.8)
    forecaster.update(0.0, torch.zeros(3, 4))
    with pytest.raises(ValueError):
        forecaster.update(0.5, torch.zeros(5, 4))


# ---- state lifecycle ----

def test_reset_makes_a_second_run_identical():
    state = SpectrumState(8, warmup_steps=3, window_size=2.0, flex_window=0.75)
    first = play(state)
    summary = state.summary()
    state.reset()
    assert state.forecasters == {}
    assert (state.ran_steps, state.skipped_steps) == (0, 0)
    second = play(state)
    assert state.summary() == summary
    assert first and len(first) == len(second)
    for (step_a, key_a, value_a), (step_b, key_b, value_b) in zip(first, second):
        assert (step_a, key_a) == (step_b, key_b)
        assert torch.equal(value_a, value_b)


def test_clear_releases_the_stored_features():
    state = SpectrumState(8, warmup_steps=3)
    for step in range(8):
        if state.should_run(step):
            state.store(0, state.time_for(step), torch.ones(4, 4))
    assert state.peak_forecaster_bytes() == 5 * 4 * 4 * 4
    state.clear()
    assert state.peak_forecaster_bytes() == 0


def test_time_for_on_both_axes():
    steps = SpectrumState(50, warmup_steps=5, time_axis="step")
    assert steps.time_for(0) == 0.0
    assert steps.time_for(49) == pytest.approx(1.0)
    sigmas = SpectrumState(50, warmup_steps=5, time_axis="sigma")
    assert sigmas.time_for(10, sigma=0.25, sigma_range=(0.0, 1.0)) == pytest.approx(0.25)
    assert sigmas.time_for(10, sigma=None, sigma_range=(0.0, 1.0)) == pytest.approx(10 / 49)
    with pytest.raises(ValueError):
        SpectrumState(50, time_axis="wallclock")


def test_validation_records_the_relative_error():
    state = SpectrumState(8, warmup_steps=3, validate=True)
    real = torch.ones(4, 4)
    assert state.log_validation(4, real * 0.9, real) == pytest.approx(0.1, rel=1e-5)
    assert "worst forecast error" in state.summary()

def test_taylor_follows_time_that_falls():
    """On the sigma axis time falls from step to step; the local extrapolation must follow it.

    It used to clamp the step between the last two points to a small positive number, so a falling
    axis made the extrapolation factor enormous: a feature exactly linear in sigma came back 2.6e5x
    off. A first-order Taylor term is exact on a linear feature whichever way time runs.
    """
    from comfy.ldm.hunyuan_image_3.spectrum import SpectrumForecaster

    forecaster = SpectrumForecaster(w=0.0, max_w=0.0)          # Taylor only
    feature = lambda s: torch.full((4, 8), 3.0) + 2.0 * s      # noqa: E731
    for sigma in (1.0, 0.9):
        forecaster.update(sigma, feature(sigma))
    predicted = forecaster.predict(0.8, skip_offset=1, window=2)
    assert torch.allclose(predicted, feature(0.8), atol=1e-5), (predicted - feature(0.8)).abs().max()


def test_streamed_fit_matches_the_stacked_solve():
    """The fit accumulates X^T H row by row instead of stacking the history in float32; same answer."""
    from comfy.ldm.hunyuan_image_3.spectrum import ChebyshevForecaster

    generator = torch.Generator().manual_seed(0)
    forecaster = ChebyshevForecaster(M=4, lam=0.1, history=12)
    for step in range(9):
        forecaster.update(step / 20, torch.randn(16, 32, generator=generator).to(torch.bfloat16))
    coef = forecaster._fit()

    t = torch.tensor(forecaster.times, dtype=torch.float32)
    X = forecaster._design(2.0 * t - 1.0)
    H = torch.stack(forecaster.rows).to(torch.float32)
    reference = torch.linalg.solve(X.T @ X + 0.1 * torch.eye(X.shape[1]), X.T @ H)
    assert torch.allclose(coef, reference, rtol=1e-4, atol=1e-5)
