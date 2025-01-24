from .view_base import BaseView

class IdentityView(BaseView):
  '''
  IdentityView: no transformations applied to image
  '''
  def __init__(self):
    super().__init__()

  def __str__(self):
        return f"Identity"

  def view(self, im):
    return im 
  
  def inverse_view(self, noise):
    return noise