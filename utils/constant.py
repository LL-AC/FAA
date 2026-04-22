
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch


IMAGE_MEAN = IMAGENET_DEFAULT_MEAN
IMAGE_STD =  IMAGENET_DEFAULT_STD


#THRESHOLD = torch.tensor([.5,.5,.7,.3,.2, .5])


PLOT_AXIS = ['no_AS', 'Early_AS', 'Significant_AS']
TABLE_CLOUMNS = ['' ,'no_AS', 'Early_AS', 'Significant_AS']
