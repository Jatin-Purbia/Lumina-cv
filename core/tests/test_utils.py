"""Tests for utils (seed, device, registry) and the shared CLI layer; TESTING strategy.

CPU only and deterministic. CUDA behaviour is exercised through monkeypatched availability
queries, plus real-device checks that are skipped when no GPU is present.
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Callable, Iterator

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from nta_zerodce.cli import (
    ARG_TO_FIELD,
    EXIT_RUNTIME,
    EXIT_USAGE,
    TRACEBACK_ENV,
    add_common_arguments,
    flag_conflict,
    main_guard,
    overrides_from_args,
)
from nta_zerodce.training.config import FeatureFlags
from nta_zerodce.utils.device import resolve_device
from nta_zerodce.utils.registry import Registry
from nta_zerodce.utils.seed import make_generator, seed_everything, worker_init_fn

# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------

_ENV_KEYS = ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")


@pytest.fixture
def restore_rng_state() -> Iterator[None]:
    """Undo the global side effects of seed_everything after a test."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    env = {key: os.environ.get(key) for key in _ENV_KEYS}
    py_state, np_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)


def _draw() -> tuple[float, float, list[float]]:
    """One draw from each global RNG."""
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("deterministic", [True, False])
def test_same_seed_same_draws(deterministic: bool) -> None:
    """Re-seeding reproduces random, numpy and torch draws exactly."""
    seed_everything(17, deterministic=deterministic)
    first = _draw()
    seed_everything(17, deterministic=deterministic)
    assert _draw() == first


@pytest.mark.usefixtures("restore_rng_state")
def test_different_seeds_differ() -> None:
    """Different seeds give different draws in all three generators."""
    seed_everything(17)
    a = _draw()
    seed_everything(29)
    b = _draw()
    assert a[0] != b[0] and a[1] != b[1] and a[2] != b[2]


@pytest.mark.usefixtures("restore_rng_state")
def test_seed_advances_streams_and_is_reset_by_reseeding() -> None:
    """Draws advance the streams; seeding again restarts them."""
    seed_everything(3)
    first, second = _draw(), _draw()
    assert first != second
    seed_everything(3)
    assert _draw() == first


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("seed", [0, 17, 2**32 - 1])
def test_valid_seed_boundaries(seed: int) -> None:
    """The documented range [0, 2**32 - 1] is accepted."""
    seed_everything(seed)


@pytest.mark.parametrize("seed", [-1, -17, 2**32, 2**40])
def test_out_of_range_seed_rejected(seed: int) -> None:
    """Negative and too-wide seeds raise ValueError before any state is touched."""
    with pytest.raises(ValueError, match="seed"):
        seed_everything(seed)


@pytest.mark.parametrize("seed", [True, 1.5, "17", None])
def test_non_integer_seed_rejected(seed: object) -> None:
    """Bools, floats, strings and None are not seeds."""
    with pytest.raises(ValueError, match="seed"):
        seed_everything(seed)  # type: ignore[arg-type]


@pytest.mark.usefixtures("restore_rng_state")
def test_deterministic_flag_switches_torch_modes() -> None:
    """deterministic=True enables deterministic algorithms and disables cudnn benchmark."""
    seed_everything(1, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    seed_everything(1, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.usefixtures("restore_rng_state")
def test_pythonhashseed_exported_for_children() -> None:
    """PYTHONHASHSEED is set (affects child processes only)."""
    seed_everything(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


@pytest.mark.usefixtures("restore_rng_state")
def test_cuda_seeding_and_cublas_config_without_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CUDA reported present, every device is seeded and CUBLAS config is set."""
    seen: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda s: seen.append(s))
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    seed_everything(9, deterministic=True)
    assert seen and set(seen) == {9}  # torch.manual_seed may also seed CUDA
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    seed_everything(9, deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"  # user value is kept
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    seed_everything(9, deterministic=False)
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ


def test_make_generator_is_seeded_and_independent() -> None:
    """Generators reproduce per seed and do not consume the global stream."""
    a = torch.rand(4, generator=make_generator(5))
    b = torch.rand(4, generator=make_generator(5))
    c = torch.rand(4, generator=make_generator(6))
    assert torch.equal(a, b) and not torch.equal(a, c)
    state = torch.get_rng_state()
    torch.rand(2, generator=make_generator(5))
    assert torch.equal(state, torch.get_rng_state())


def test_make_generator_device_argument() -> None:
    """The device may be a string or torch.device; invalid seeds are rejected."""
    assert make_generator(1, "cpu").device.type == "cpu"
    assert make_generator(1, torch.device("cpu")).device.type == "cpu"
    with pytest.raises(ValueError, match="seed"):
        make_generator(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_make_generator_on_cuda() -> None:
    """A CUDA generator reproduces draws on the device."""
    a = torch.rand(3, device="cuda", generator=make_generator(2, "cuda"))
    b = torch.rand(3, device="cuda", generator=make_generator(2, "cuda"))
    assert torch.equal(a, b)


@pytest.mark.usefixtures("restore_rng_state")
def test_worker_init_fn_mirrors_torch_seed_into_python_and_numpy() -> None:
    """worker_init_fn seeds random and numpy from torch.initial_seed()."""
    torch.manual_seed(5)
    worker_init_fn(0)
    expected = (random.random(), float(np.random.rand()))
    random.seed(5)
    np.random.seed(5)
    assert expected == (random.random(), float(np.random.rand()))


class _RandomItems(Dataset[tuple[float, float, float]]):  # type: ignore[misc]
    """Dataset whose items expose all three RNGs, to check worker seeding."""

    def __len__(self) -> int:
        """Four items."""
        return 4

    def __getitem__(self, index: int) -> tuple[float, float, float]:
        """Return one draw from python, numpy and torch."""
        return random.random(), float(np.random.rand()), float(torch.rand(1))


def _load(seed: int) -> list[list[float]]:
    """Run a 2-worker DataLoader with a seeded generator and flatten the batches."""
    loader = DataLoader(
        _RandomItems(),
        batch_size=1,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        generator=make_generator(seed),
    )
    return [[float(x) for x in batch] for batch in loader]


def test_dataloader_workers_are_reproducible_and_distinct() -> None:
    """Same generator seed reproduces worker draws; workers do not share a stream."""
    first = _load(7)
    assert _load(7) == first
    assert _load(8) != first
    # Items 0 and 1 come from different workers; python/numpy/torch draws all differ.
    assert all(a != b for a, b in zip(first[0], first[1], strict=True))


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------


def test_cpu_and_default_auto() -> None:
    """'cpu' is CPU; the default argument is 'auto'."""
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device() == resolve_device("auto")


def test_auto_follows_cuda_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto picks cuda when available and cpu otherwise."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


@pytest.mark.parametrize("name", ["cuda", "cuda:0", "cuda:3"])
def test_explicit_cuda_without_cuda_raises(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Explicit CUDA never silently falls back to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device(name)


def test_cuda_index_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Indices below device_count resolve; at or above raise RuntimeError."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device("cuda") == torch.device("cuda")
    assert resolve_device("cuda:0") == torch.device("cuda:0")
    assert resolve_device("cuda:1") == torch.device("cuda:1")
    with pytest.raises(RuntimeError, match="out of range"):
        resolve_device("cuda:2")
    with pytest.raises(RuntimeError, match="out of range"):
        resolve_device("cuda:99")


@pytest.mark.parametrize(
    "name",
    ["gpu", "CUDA", "CPU", "Auto", "cuda:", "cuda:-1", "cuda:x", "cuda:1:2", "mps", "", " cpu"],
)
def test_invalid_device_strings(name: str) -> None:
    """Anything outside auto/cpu/cuda/cuda:N is a ValueError (not a RuntimeError)."""
    with pytest.raises(ValueError, match="device"):
        resolve_device(name)


def test_non_string_device() -> None:
    """Non-strings are rejected."""
    with pytest.raises(ValueError, match="device"):
        resolve_device(0)  # type: ignore[arg-type]


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present")
def test_real_machine_without_cuda_rejects_cuda() -> None:
    """On a CPU-only machine explicit cuda fails and auto gives cpu."""
    with pytest.raises(RuntimeError):
        resolve_device("cuda")
    assert resolve_device("auto") == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_real_machine_with_cuda() -> None:
    """On a GPU machine cuda, cuda:0 resolve and an absurd index fails."""
    assert resolve_device("cuda").type == "cuda"
    assert resolve_device("cuda:0").index == 0
    with pytest.raises(RuntimeError):
        resolve_device(f"cuda:{torch.cuda.device_count()}")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_register_get_contains_len_names() -> None:
    """Basic registry life cycle with sorted names."""
    registry: Registry[int] = Registry()
    assert len(registry) == 0 and registry.names() == [] and "a" not in registry
    registry.register("b", 2)
    registry.register("a", 1)
    assert registry.get("a") == 1 and registry.get("b") == 2
    assert len(registry) == 2 and "a" in registry and "z" not in registry
    assert registry.names() == ["a", "b"]


def test_registry_names_returns_a_copy() -> None:
    """Mutating the returned list does not change the registry."""
    registry: Registry[int] = Registry()
    registry.register("a", 1)
    registry.names().append("hacked")
    assert registry.names() == ["a"]


def test_registry_duplicate_name_rejected() -> None:
    """Duplicate registration fails fast and keeps the first value."""
    registry: Registry[str] = Registry()
    registry.register("x", "first")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("x", "second")
    assert registry.get("x") == "first" and len(registry) == 1


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_registry_empty_name_rejected(name: str) -> None:
    """Empty or blank names are rejected."""
    registry: Registry[int] = Registry()
    with pytest.raises(ValueError, match="nonempty"):
        registry.register(name, 1)
    assert len(registry) == 0


@pytest.mark.parametrize("name", [None, 3, b"x"])
def test_registry_non_string_name_rejected(name: object) -> None:
    """Non-string names are rejected with ValueError."""
    registry: Registry[int] = Registry()
    with pytest.raises(ValueError):
        registry.register(name, 1)  # type: ignore[arg-type]


def test_registry_unknown_name_raises_key_error_listing_known() -> None:
    """Unknown names raise KeyError and the message lists registered names."""
    registry: Registry[int] = Registry()
    registry.register("alpha", 1)
    with pytest.raises(KeyError, match="beta") as info:
        registry.get("beta")
    assert "alpha" in str(info.value)
    with pytest.raises(KeyError):
        Registry[int]().get("anything")


def test_registry_stores_callables_and_is_per_instance() -> None:
    """Values may be callables; registries do not share state."""
    one: Registry[Callable[[int], int]] = Registry()
    two: Registry[Callable[[int], int]] = Registry()
    one.register("double", lambda x: 2 * x)
    assert one.get("double")(4) == 8
    assert "double" not in two
    assert "double" not in Registry[int]()


def test_registry_contains_tolerates_unhashable_probe_types() -> None:
    """`in` works with non-string probes without raising for hashable objects."""
    registry: Registry[int] = Registry()
    registry.register("a", 1)
    assert 5 not in registry and None not in registry


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

EXPECTED_OPTIONS = {
    "--config",
    "--noise",
    "--no-noise",
    "--denoise",
    "--no-denoise",
    "--edge",
    "--no-edge",
    "--chroma",
    "--no-chroma",
    "--adaptive-exp",
    "--no-adaptive-exp",
    "--temporal",
    "--no-temporal",
    "--light",
    "--no-light",
    "--curve-scale",
    "--video-dir",
    "--seed",
    "--device",
}


def _parser() -> argparse.ArgumentParser:
    """Parser holding only the shared flags."""
    parser = argparse.ArgumentParser(prog="t")
    add_common_arguments(parser)
    return parser


def test_common_arguments_are_exactly_the_contract() -> None:
    """The parser exposes exactly the INTERFACES flags (and -h/--help)."""
    options = {opt for action in _parser()._actions for opt in action.option_strings}
    assert options - {"-h", "--help"} == EXPECTED_OPTIONS


def test_defaults_are_absent() -> None:
    """Everything defaults to None except --config."""
    args = vars(_parser().parse_args([]))
    assert args.pop("config") == "configs/baseline.yaml"
    assert args and all(value is None for value in args.values())


def test_all_flags_parse_with_types() -> None:
    """Each flag parses to its documented type."""
    args = _parser().parse_args(
        [
            "--config",
            "c.yaml",
            "--noise",
            "--no-denoise",
            "--edge",
            "--no-chroma",
            "--adaptive-exp",
            "--no-temporal",
            "--light",
            "--curve-scale",
            "0.5",
            "--video-dir",
            "vids",
            "--seed",
            "29",
            "--device",
            "cuda:1",
        ]  # fmt: skip
    )
    assert args.config == "c.yaml"
    assert (args.noise, args.denoise, args.edge, args.chroma) == (True, False, True, False)
    assert (args.adaptive_exp, args.temporal, args.light) == (True, False, True)
    assert args.curve_scale == 0.5 and isinstance(args.curve_scale, float)
    assert args.video_dir == "vids" and args.seed == 29 and isinstance(args.seed, int)
    assert args.device == "cuda:1"


@pytest.mark.parametrize(
    "argv", [["--seed", "x"], ["--curve-scale", "fast"], ["--seed", "1.5"], ["--bogus"]]
)
def test_bad_command_lines_exit_with_usage_error(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """argparse rejects bad values with exit status 2."""
    with pytest.raises(SystemExit) as info:
        _parser().parse_args(argv)
    assert info.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_adding_common_arguments_twice_conflicts() -> None:
    """Defining the shared flags on a parser that already has them is an error."""
    parser = _parser()
    with pytest.raises(argparse.ArgumentError):
        add_common_arguments(parser)


def test_script_specific_arguments_can_coexist() -> None:
    """Scripts may add their own options next to the shared ones."""
    parser = _parser()
    parser.add_argument("--resume", default=None)
    assert parser.parse_args(["--resume", "ck.pt", "--noise"]).resume == "ck.pt"


def test_overrides_keys_and_absent_values() -> None:
    """All eleven dotted keys are always present; absent flags are None, not False."""
    overrides = overrides_from_args(_parser().parse_args([]))
    assert set(overrides) == {
        "flags.noise_aware",
        "flags.denoise",
        "flags.edge",
        "flags.chroma",
        "flags.adaptive_exp",
        "flags.temporal",
        "flags.light",
        "model.curve_scale",
        "paths.video_dir",
        "training.seed",
        "training.device",
    }
    assert all(value is None for value in overrides.values())
    assert overrides == overrides_from_args(argparse.Namespace())  # missing attrs -> None


@pytest.mark.parametrize(
    ("argv", "key", "value"),
    [
        (["--noise"], "flags.noise_aware", True),
        (["--no-noise"], "flags.noise_aware", False),
        (["--denoise"], "flags.denoise", True),
        (["--no-edge"], "flags.edge", False),
        (["--chroma"], "flags.chroma", True),
        (["--adaptive-exp"], "flags.adaptive_exp", True),
        (["--no-adaptive-exp"], "flags.adaptive_exp", False),
        (["--temporal"], "flags.temporal", True),
        (["--no-light"], "flags.light", False),
        (["--curve-scale", "0.25"], "model.curve_scale", 0.25),
        (["--video-dir", "d"], "paths.video_dir", "d"),
        (["--seed", "0"], "training.seed", 0),
        (["--device", "cpu"], "training.device", "cpu"),
    ],
)
def test_overrides_mapping(argv: list[str], key: str, value: object) -> None:
    """Each flag maps to its dotted field; explicit false and zero are kept (not None)."""
    overrides = overrides_from_args(_parser().parse_args(argv))
    assert overrides[key] == value and type(overrides[key]) is type(value)
    assert {k: v for k, v in overrides.items() if k != key} == {
        k: None for k in overrides if k != key
    }


def test_arg_to_field_table_matches_parser_destinations() -> None:
    """The public mapping covers exactly the parser destinations except --config."""
    destinations = {action.dest for action in _parser()._actions} - {"help", "config"}
    assert set(ARG_TO_FIELD) == destinations


def test_flag_conflict_messages() -> None:
    """Only denoise-without-noise is a conflict; the message blames --no-noise when given."""
    assert flag_conflict(FeatureFlags()) is None
    assert flag_conflict(FeatureFlags(noise_aware=True, denoise=True)) is None
    assert flag_conflict(FeatureFlags(noise_aware=True)) is None
    plain = flag_conflict(FeatureFlags(denoise=True))
    assert plain is not None and "denoise" in plain and "was given" not in plain
    blamed = flag_conflict(FeatureFlags(denoise=True), _parser().parse_args(["--no-noise"]))
    assert blamed is not None and "--no-noise was given" in blamed and "\n" not in blamed
    other = flag_conflict(FeatureFlags(denoise=True), _parser().parse_args(["--denoise"]))
    assert other is not None and "was given" not in other


# main_guard ----------------------------------------------------------------


def _raiser(exc: BaseException) -> Callable[[], int | None]:
    """Build a run() that raises exc."""

    def run() -> int | None:
        """Raise the stored exception."""
        raise exc

    return run


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (ValueError("bad value"), EXIT_USAGE),
        (FileNotFoundError("missing thing"), EXIT_USAGE),
        (NotImplementedError("not yet"), EXIT_RUNTIME),
        (RuntimeError("cuda gone"), EXIT_RUNTIME),
        (OSError("disk"), EXIT_RUNTIME),
    ],
)
def test_main_guard_maps_exceptions_to_exit_codes(
    exc: BaseException, code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """Expected exceptions give one stderr line, no traceback, and the documented exit code."""
    assert main_guard(_raiser(exc), prog="tool") == code
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1 and lines[0] == f"tool: error: {exc}"
    assert "Traceback" not in captured.err


def test_main_guard_message_is_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    """Multi-line messages are collapsed; empty messages fall back to the type name."""
    assert main_guard(_raiser(ValueError("a\n  b\n\nc")), prog="t") == EXIT_USAGE
    assert capsys.readouterr().err == "t: error: a b c\n"
    assert main_guard(_raiser(ValueError()), prog="t") == EXIT_USAGE
    assert capsys.readouterr().err == "t: error: ValueError\n"


def test_main_guard_success_codes() -> None:
    """None means success; returned integers pass through."""
    assert main_guard(lambda: None) == 0
    assert main_guard(lambda: 0) == 0
    assert main_guard(lambda: 3) == 3


def test_main_guard_passes_through_system_exit_and_unexpected_errors() -> None:
    """argparse's SystemExit and genuine bugs are not swallowed."""
    with pytest.raises(SystemExit) as info:
        main_guard(_raiser(SystemExit(2)))
    assert info.value.code == 2
    with pytest.raises(KeyError):
        main_guard(_raiser(KeyError("bug")))
    with pytest.raises(ZeroDivisionError):
        main_guard(lambda: 1 // 0)


def test_main_guard_keyboard_interrupt(capsys: pytest.CaptureFixture[str]) -> None:
    """Ctrl-C returns 130 with a short message."""
    assert main_guard(_raiser(KeyboardInterrupt()), prog="t") == 130
    assert "interrupted" in capsys.readouterr().err


def test_main_guard_traceback_env_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    """LUMINA_TRACEBACK=1 re-raises for debugging; unset restores the one-line behaviour."""
    monkeypatch.setenv(TRACEBACK_ENV, "1")
    with pytest.raises(ValueError, match="boom"):
        main_guard(_raiser(ValueError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        main_guard(_raiser(RuntimeError("boom")))
    monkeypatch.delenv(TRACEBACK_ENV)
    assert main_guard(_raiser(ValueError("boom")), prog="t") == EXIT_USAGE


def test_main_guard_default_prog_uses_argv0(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without prog the basename of sys.argv[0] prefixes the message."""
    monkeypatch.setattr("sys.argv", ["/x/y/train.py"])
    main_guard(_raiser(ValueError("oops")))
    assert capsys.readouterr().err == "train.py: error: oops\n"


def test_main_guard_end_to_end_with_config_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A real config failure through resolve_config becomes exit 2 with a readable line."""
    from nta_zerodce.cli import resolve_config

    def run() -> int:
        """Resolve full.yaml with noise switched off (denoise stays on)."""
        resolve_config(_parser().parse_args(["--config", "configs/full.yaml", "--no-noise"]))
        return 0

    assert main_guard(run, prog="train.py") == EXIT_USAGE
    err = capsys.readouterr().err
    assert err.startswith("train.py: error: denoise requires") and err.count("\n") == 1
