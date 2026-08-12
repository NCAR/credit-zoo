"""credit-zoo: community/reference model architectures for CREDIT.

Importing this package does nothing by itself. Each model under
``credit_zoo.models.<name>`` is a plain ``credit.models.base_model.BaseModel``
subclass with no CREDIT-specific decoration. Models register into CREDIT's
model factory through a training config's ``custom_objects:`` block
(``credit.registry.load_custom_objects``), which imports the module and
registers the class by dotted path. See the top-level README for usage.
"""
