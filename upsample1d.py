import copy
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Upsample1d(torch.nn.Module):
    def __init__(self, scale = 2):
        super(Upsample1d, self).__init__()
        #self.up = torch.nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        self.scale = scale
    def forward(self, forward_input):
    	#print("before unsqueeze ", forward_input.size())
    	#n,c,l = forward_input.size()
    	#forward_input = torch.unsqueeze(forward_input, dim=3)
    	#print("after unsqueeze ", forward_input.size())
    	
    	#forward_input = self.up(forward_input)
    	#print("after upsample ", forward_input.size())
    	#forward_input = forward_input.permute(0,3,1,2)
    	#forward_input = forward_input[:,0,:,:]
    	#forward_input = torch.squeeze(forward_input)
    	#print("after squeeze ", forward_input.size())
    	forward_input = torch.nn.functional.interpolate(forward_input, scale_factor = self.scale, mode='nearest')
    	return forward_input
