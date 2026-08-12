"""credit-zoo: community/reference model architectures for CREDIT.

Importing this package does nothing by itself. Each model under
``credit_zoo.models.<name>`` self-registers into CREDIT's model factory via
``credit.models.register_model`` when its module is imported — either
directly (``import credit_zoo.models.aurora.model``) or through a config's
``custom_models:`` list. See the top-level README for usage.
"""
