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
`credit.registry.load_custom_objects` (the config-level `custom_objects:` block). Model classes here are
plain `credit.models.base_model.BaseModel` subclasses with no CREDIT-specific decoration at all.

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

`credit` resolves `model.type` through its internal registry, and `load_model` calls
`credit.registry.load_custom_objects(conf)` before dispatching — so any model listed under a config's
`custom_objects:` block gets imported and registered by dotted path, with zero changes to `credit` itself
and no decorator on the model class. Add one entry per zoo model you want to use in that run:

```yaml
# my_config.yml
custom_objects:
  aurora:
    object_type: model
    module_path: credit_zoo.models.aurora.model
    module_name: CREDITAurora

save_loc: '/glade/derecho/scratch/$USER/CREDIT_runs/aurora'

model:
  type: "aurora"
  surf_vars: ["2t", "10u", "10v", "msl"]
  atmos_vars: ["z", "u", "v", "t", "q"]
  atmos_levels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  # ... remaining model/data/trainer config, see config/aurora.yml for a full example
```

Requires `credit_zoo` to be installed (`pip install -e .` here, or a normal `pip install`) so
`module_path` resolves as an ordinary import — `custom_objects` does not accept file paths, only
importable dotted paths. If you're calling `load_model` from your own script instead of the `credit` CLI,
`load_custom_objects` still does the work as long as it runs before `load_model`'s registry lookup (it
does, internally); you never need to import the zoo model module yourself.

See `credit/registry.py::load_custom_objects` in `miles-credit` for the full mechanism (also covers
datasets, preblocks, postblocks, losses, and metrics), and the [`config/`](config) directory here for a
complete example config per model — each already includes its `custom_objects:` block.

## Repo layout

- `credit_zoo/models/<name>/` — one directory per model (`__init__.py`, implementation file(s), usually a
  `README.md`). `credit_zoo/models/MODELS.md` is the full model-by-model reference.
- `config/` — per-model training configs (`config/<name>.yml`), moved verbatim from the original PR and
  updated to carry the `custom_objects:` block described above, so each runs as-is.
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
