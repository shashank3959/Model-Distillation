from .search_model_darts_v1 import TinyNetworkDartsV1
from .search_model_darts_v2 import TinyNetworkDartsV2
from .search_model_gdas     import TinyNetworkGDAS
from .genotypes             import Structure as CellStructure, architectures as CellArchitectures

nas_super_nets = {'DARTS-V1': TinyNetworkDartsV1,
                  'DARTS-V2': TinyNetworkDartsV2,
                  'GDAS'    : TinyNetworkGDAS}
