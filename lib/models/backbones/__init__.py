
from .hrnet.moc_backbone import MocBackbone
from .hrnet.seg_hrnet_hloc import MocHRBackbone
from .hrnet.seg_hrnet_fpn import HRBackboneFPN
from .hrnet.seg_hrnet_cat import MocCatBackbone
from .maevit.vitdet import MAEvitBackbone
from .vgg.vgg import VGGBackbone
from .mobilenet.mobilenet import MobileBackbone
__all__ = [
     "MocBackbone","MocHRBackbone","MocCatBackbone",
    'MAEvitBackbone', "VGGBackbone","HRBackboneFPN",
    "MobileBackbone"
]