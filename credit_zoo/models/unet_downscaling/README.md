# Downscaling U-Net (`unet_downscaling`)

The [`unet`](../unet/README.md) segmentation-models-pytorch U-Net adapted for regional downscaling: inputs are the
`boundary` + `prognostic` channels and outputs are the `prognostic` + `diagnostic` channels, given as a
`channels:` dict.

Moved here from `credit.models.unet_downscaling` in miles-credit (NCAR/miles-credit#519). The class and its
arguments are unchanged.

```yaml
custom_objects:
  unet_downscaling:
    object_type: model
    module_path: credit_zoo.models.unet_downscaling.unet_downscaling
    module_name: DownscalingSegmentationModel
```
