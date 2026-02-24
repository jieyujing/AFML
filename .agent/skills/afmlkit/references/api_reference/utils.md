# API Reference: utils.py

**Language**: Python

**Source**: `feature\utils.py`

---

## Classes

### ComputationGraph

Directed acyclic graph (DAG) capturing feature dependencies.

**Inherits from**: (none)

#### Methods

##### __init__(self)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |


##### add_edge(self, src: str, dst: str)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| src | str | - | - |
| dst | str | - | - |


##### add_node(self, node: str)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |
| node | str | - | - |


##### topological_sort(self) → List[str]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `List[str]`


##### visualize(self) → str

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| self | None | - | - |

**Returns**: `str`




## Functions

### _serialize_value(val: Any) → Any

Best-effort JSON-serializable conversion for common types used in transforms.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| val | Any | - | - |

**Returns**: `Any`



### _deserialize_value(val: Any) → Any

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| val | Any | - | - |

**Returns**: `Any`



### _class_path(obj: Any) → str

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| obj | Any | - | - |

**Returns**: `str`



### _import_class(path: str)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| path | str | - | - |

**Returns**: (none)



### _maybe_unary_from_name(name: str)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| name | str | - | - |

**Returns**: (none)



### transform_to_config(t: BaseTransform) → Dict[str, Any]

Serialize a BaseTransform (including operation and Compose transforms) into a dict.

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| t | BaseTransform | - | - |

**Returns**: `Dict[str, Any]`



### transform_from_config(cfg: Dict[str, Any]) → BaseTransform

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| cfg | Dict[str, Any] | - | - |

**Returns**: `BaseTransform`



### _flatten_requires(t: BaseTransform) → List[str]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| t | BaseTransform | - | - |

**Returns**: `List[str]`



### build_feature_graph(features: List['Feature']) → ComputationGraph

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| features | List['Feature'] | - | - |

**Returns**: `ComputationGraph`



### _child_out_names(t: BaseTransform) → List[str]

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| t | BaseTransform | - | - |

**Returns**: `List[str]`


