from typing import Dict, Tuple, Optional, NamedTuple, Union
from torch import Tensor

try:
  from typing import Literal
except ImportError:
  from typing_extensions import Literal

BoundingBox = Tuple[float, float, float, float] 
SplitType = Literal['train', 'validation', 'test']

class Category(NamedTuple):
    id: str
    super_category: Optional[str]
    name: str


class Annotation(NamedTuple):
    area: float
    image_id: str
    bbox: BoundingBox
    category_no: int
    category_id: str
    id: Optional[int] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    is_group_of: Optional[bool] = None
    is_truncated: Optional[bool] = None
    is_occluded: Optional[bool] = None
    is_depiction: Optional[bool] = None
    is_inside: Optional[bool] = None
    segmentation: Optional[Dict] = None
