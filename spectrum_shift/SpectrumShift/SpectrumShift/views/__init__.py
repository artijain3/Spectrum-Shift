from .view_identity import IdentityView
from .view_colorblind import ColorblindView

class Views:
  IDENTITY = 0
  COLORBLIND = 1

  VIEW_MAPPING = {IDENTITY: IdentityView,
                  COLORBLIND: ColorblindView}

def get_views(view_names):
  return [Views.VIEW_MAPPING[name]() for name in view_names] 