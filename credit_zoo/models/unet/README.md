# U-Net (`unet`)

A convolutional U-Net built on [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
(smp). It is a pure convolutional encoder-decoder with skip connections and no attention, so it is fast,
memory-light, and a simple baseline. The encoder can be any backbone smp supports (e.g. a ResNet), optionally with
pretrained ImageNet weights. `rk4_integration: True` wraps the network in a 4th-order Runge-Kutta step.

Moved here from `credit.models.unet` in miles-credit so that core CREDIT no longer depends on
segmentation-models-pytorch (NCAR/miles-credit#519). The class and its arguments are unchanged, so existing
checkpoints load as before.

```yaml
custom_objects:
  unet:
    object_type: model
    module_path: credit_zoo.models.unet.unet
    module_name: SegmentationModel

model:
  type: "unet"
  image_height: 192
  image_width: 288
  frames: 1
  levels: 16
  channels: 4
  surface_channels: 7
  input_only_channels: 3
  output_only_channels: 0
  rk4_integration: False
  architecture:
    name: "unet"              # unet, unet++, manet, linknet, fpn, pspnet, pan, deeplabv3, deeplabv3+
    encoder_name: "resnet34"  # any smp encoder
    encoder_weights: "imagenet"
```

See [`config/unet.yml`](../../../config/unet.yml) for a complete example.
