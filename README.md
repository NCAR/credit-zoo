# credit-zoo

A collection of community/reference model architectures for [CREDIT](https://github.com/NCAR/miles-credit)
(Community Research Earth Digital Intelligence Twin), NSF NCAR's platform for training and deploying AI
atmospheric prediction models. This repo was extracted from `miles-credit` PR #312 ("model-zoo") so the
zoo models can be maintained and versioned independently of CREDIT core.

It currently ships 19 model architectures ported into CREDIT's `(B, C, H, W)` / `(B, C, T, H, W)` tensor
interface: `ace`, `aifs`, `arches`, `aurora`, `climax`, `corrdiff`, `dlesym`, `fengwu`, `fourcastnet`,
`fourcastnet3`, `fuxi_ens`, `graphcast`, `healpix`, `itransformer`, `mambavision`, `pangu`, `sfno`,
`stormer`, `swinrnn`. See [`credit_zoo/models/MODELS.md`](credit_zoo/models/MODELS.md) for the full
reference (distributed-training support, pretrained-weight compatibility, licenses, and per-model
architecture notes) and [`docs/models.md`](docs/models.md) for `torch.compile` compatibility notes.

This repo does not vendor or modify any CREDIT core code. It depends on `credit` (the `miles-credit`
package) as an ordinary pip dependency and plugs into it purely through CREDIT's public extension point,
`credit.models.register_model`.

## Install

```bash
pip install -e .
```

This pulls in `credit` (from `git+ssh://git@github.com/NCAR/miles-credit.git`) plus the third-party
packages the zoo models need (`torch_geometric`, `torch-harmonics`, `timm`, `healpy`, `scipy`, ...). A few
per-model accelerators (`flash_attn`, `mamba_ssm`, `torch_scatter`, `scikit-learn`, `xarray`) are optional
— every model falls back to a pure-PyTorch path (or a clear error naming the missing package) when they
are absent — and are grouped under the `extras` optional-dependency:

```bash
pip install -e ".[extras]"
```

## Usage

Each model class self-registers into CREDIT's model factory via the `@register_model` decorator, so
nothing in `credit` itself needs to change. The recommended way to use a zoo model with the `credit` CLI
is to point a config's `custom_models:` list at the model's source file — CREDIT imports it (which runs
the decorator, registering the model type) before `load_model` runs:

```yaml
# my_config.yml
custom_models:
  - '/path/to/credit-zoo/credit_zoo/models/aurora/model.py'

save_loc: '/glade/derecho/scratch/$USER/CREDIT_runs/aurora'

model:
  type: "aurora"
  surf_vars: ["2t", "10u", "10v", "msl"]
  atmos_vars: ["z", "u", "v", "t", "q"]
  atmos_levels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  # ... remaining model/data/trainer config, see config/aurora.yml for a full example
```

`custom_models` paths go through `os.path.expandvars`, so `$WORK`/`$USER`-style variables work, but the
path is otherwise resolved as given — use an absolute path (or a path relative to wherever `credit` is
invoked from). If you're calling `load_model` from your own script instead of the CLI, importing the
module directly has the same effect:

```python
import credit_zoo.models.aurora.model  # registers "aurora" as a side-effect
from credit.models import load_model

model = load_model(conf)  # conf["model"]["type"] == "aurora"
```

See `credit/models/__init__.py::register_model` in `miles-credit` for the full decorator contract, and
the [`config/`](config) directory here for a complete example config per model.

## Repo layout

- `credit_zoo/models/<name>/` — one directory per model (`__init__.py`, implementation file(s), usually a
  `README.md`). `credit_zoo/models/MODELS.md` is the full model-by-model reference.
- `config/` — per-model training configs (`config/<name>.yml`), moved verbatim from the original PR.
  **Note:** these configs use `model: type: <key>` alone; add a `custom_models:` entry (see above) before
  running one, since the type key is not registered until the corresponding model file is imported.
- `config/smoke/` — smaller smoke-test configs and shell launchers (`run_all_local.sh`,
  `submit_all_casper.sh`, `submit_noinit_casper.sh`, `submit_pretrained_casper.sh`) for quick
  forward/backward-pass validation on NCAR HPC.
- `tests/` — `test_model_zoo.py` (forward/backward smoke test for every model) and
  `test_model_zoo_train.py` (a short real training loop per model).
- `scripts/` — `compute_wb2_stats.py` / `submit_wb2_stats.sh`, WeatherBench2 verification statistics.
- `docs/models.md` — `torch.compile` compatibility notes for the zoo models.

## License

Apache-2.0 (see `LICENSE`), matching `miles-credit`. Several individual model ports carry their own
upstream license note in their model directory — see `credit_zoo/models/MODELS.md`'s "Credits & Licenses"
table for the full per-model breakdown (several are MIT/BSD/Apache-2.0; a couple, like `pangu`'s original
weights, are non-commercial-only).
