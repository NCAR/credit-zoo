# credit-zoo

## About

credit-zoo is a collection of community/reference model architectures for [CREDIT](https://github.com/NCAR/miles-credit)
(Community Research Earth Digital Intelligence Twin), NSF NCAR's open-source platform for training and deploying
AI atmospheric prediction models. It ships CREDIT-compatible ports of published architectures — Aurora, Pangu-Weather,
GraphCast, FourCastNet, ClimaX, and more — so they can be trained and evaluated through the same `credit` CLI and
config pipeline as CREDIT's built-in models, without living in CREDIT core.

credit-zoo does not vendor or modify any CREDIT code. It depends on `credit` (the `miles-credit` package) as an
ordinary pip dependency and plugs into it through CREDIT's public extension point,
[`custom_objects`](https://miles-credit.readthedocs.io/en/latest/) — every model here is a plain
`credit.models.base_model.BaseModel` subclass with no CREDIT-specific decoration.

**New to CREDIT itself?** See the [CREDIT Quickstart](https://miles-credit.readthedocs.io/en/latest/quickstart.html)
first — credit-zoo assumes a working `credit` install.

## Getting started

```bash
pip install -e .
```

installs `credit_zoo` plus `credit` (from `git+ssh://git@github.com/NCAR/miles-credit.git`) and the third-party
packages the zoo models need. A few per-model accelerators (`flash_attn`, `mamba_ssm`, `torch_scatter`,
`scikit-learn`, `xarray`) are optional — every model falls back to a pure-PyTorch path when they're absent — and
live under the `extras` group:

```bash
pip install -e ".[extras]"
```

Every model in [`config/`](config) already ships a runnable example config with its `custom_objects:` block
pre-filled, so training one is the same as training any CREDIT model:

```bash
credit check -c config/aurora.yml   # validate first
credit train -c config/aurora.yml
```

To wire a zoo model into your own config, add one `custom_objects` entry per model you use:

```yaml
custom_objects:
  aurora:
    object_type: model
    module_path: credit_zoo.models.aurora.model
    module_name: CREDITAurora

model:
  type: "aurora"
  # ... see config/aurora.yml for a complete example
```

`credit`'s `load_model` calls `credit.registry.load_custom_objects(conf)` before dispatching, so this is all that's
needed — no changes to `credit` itself, and the module only needs to be *importable* (`credit_zoo` installed), not
imported by hand.

## Available models

`ace`, `aifs`, `arches`, `aurora`, `climax`, `corrdiff`, `dlesym`, `fengwu`, `fourcastnet`, `fourcastnet3`,
`fuxi_ens`, `graphcast`, `healpix`, `itransformer`, `mambavision`, `pangu`, `sfno`, `stormer`, `swinrnn`, `unet`,
`unet_downscaling` — 21 architectures ported into CREDIT's `(B, C, H, W)` / `(B, C, T, H, W)` tensor interface. See
[`credit_zoo/models/MODELS.md`](credit_zoo/models/MODELS.md) for the full per-model reference (papers, source
links, pretrained-weight availability, and architecture notes).

## Documentation

- [`credit_zoo/models/MODELS.md`](credit_zoo/models/MODELS.md) — per-model reference and the guide for adding a
  new model.
- [`docs/models.md`](docs/models.md) — `torch.compile` compatibility notes.
- `config/smoke/` — smoke-test configs and HPC launch scripts (`submit_all_casper.sh`, `submit_all_derecho.sh`)
  for a quick forward/backward-pass check of every model.
- The full [CREDIT documentation](https://miles-credit.readthedocs.io/en/latest/) covers the `custom_objects`
  registry mechanism, the gen2 config schema, and everything else this repo builds on.

## License

Apache-2.0 (see `LICENSE`), matching `miles-credit`. Several individual model ports carry their own upstream
license note — see the "Credits & Licenses" table in
[`credit_zoo/models/MODELS.md`](credit_zoo/models/MODELS.md) (most are MIT/BSD/Apache-2.0; a few, like Pangu's
original weights, are non-commercial-only).
