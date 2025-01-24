class BaseView:
    '''
    BaseView class, from which all views inherit. Implements the 
        following functions:
    '''

    def __init__(self):
        pass

    def view(self, im):
        '''
        Apply transform to an image.

        im (`torch.tensor`):
            For stage 1: Tensor of shape (3, H, W) representing a noisy image
            OR
            For stage 2: Tensor of shape (6, H, W) representing a noisy image
            concatenated with an upsampled conditioning image from stage 1
        '''
        raise NotImplementedError()

    def inverse_view(self, noise):
        '''
        Apply inverse transform to noise estimates.
            Because DeepFloyd estimates the variance in addition to
            the noise, this function must apply the inverse to the
            variance as well.

        noise (`torch.tensor`):
            Tensor of shape (6, H, W) representing the noise estimate
            (first three channel dims) and variance estimates (last
            three channel dims)
        '''
        raise NotImplementedError()