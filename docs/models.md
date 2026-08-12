# credit-zoo Model Reference

This page is a short pointer page. The full model-by-model reference — distributed training support,
pretrained-weight compatibility, licenses, and detailed architecture notes for every model in this
repo — lives in [`credit_zoo/models/MODELS.md`](../credit_zoo/models/MODELS.md).

The notes below cover `torch.compile` compatibility for the zoo models, carried over from the original
`model-zoo` PR against `miles-credit` (this content is specific to the zoo models and was not already
documented on `miles-credit` main).

## `torch.compile` compatibility

Not all models are compatible with `torch.compile` out of the box. The main blocker is
`torch.nn.utils.spectral_norm`, used by several models to stabilize long autoregressive rollouts.

| Model | Compiles? | What to do |
|-------|:---------:|------------|
| `dlesym` | Yes | Works out of the box |
| `sfno` / `fourcastnet3` / `ace` | Partial | Don't install `torch-harmonics`; the rfft2 fallback compiles cleanly (the Spherical Harmonic Transform CUDA kernels cause graph breaks) |
| `graphcast` | Partial | Use `torch.compile(model, dynamic=True)` — the icosahedral graph uses `scatter_add_` with dynamic edge counts, which otherwise triggers excessive recompilation |
| All other zoo models | Yes | Works out of the box |

### Why spectral norm blocks compilation

`spectral_norm` registers a forward hook that recomputes a normalized weight on every call.
`torch.compile`'s graph tracer cannot trace through these hooks. Spectral norm is used intentionally
to prevent rollout explosions; do not remove it without testing long rollouts.

### SFNO / FourCastNet3 / ACE

When `torch-harmonics` is installed, these models use Spherical Harmonic Transform CUDA kernels that
cause graph breaks under `torch.compile`. To use the rfft2 fallback (which compiles cleanly), don't
install `torch-harmonics`.
