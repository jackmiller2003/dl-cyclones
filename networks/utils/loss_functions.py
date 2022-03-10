import torch
import math

class L2_Dist_Func_Intensity(torch.nn.Module):
    def __init__(self):
        super(L2_Loss, self).__init__()
    
    def forward(self, pred_location:torch.Tensor, true_location: torch.Tensor, intensity: torch.Tensor, intensity_func=linear_func, intensity_scale=1000):
        """
        Here we have two inputs:
            * Predicted location -> (lon, lat, intensity)
            * True location -> (lon, lat, intensity)
        """

        R = 6371e3

        lon0, lat0 = true_location[0], true_location[1]
        lon1,lat1 = pred_location[0], pred_location[1]

        phi0 = lat0 * (math.pi/180)
        phi1 = lat1 * (math.pi/180)

        delta_phi = (lat1 - lat0) * (math.pi/180)
        delta_lambda = (lon1 - lon0) * (math.pi/180)

        a = torch.pow(torch.sin(delta_phi/2),2) + torch.cos(phi0) + torch.cos(phi1) + torch.pow(torch.sin(delta_lambda/2),2)
        a = a.float()
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        c = c.double()
        cyclone_dist = R * c

        return torch.sum(torch.pow(cyclone_dist, 2), intensity_scale * intensity_func(intensity) * intensity) / (pred_location.shape[0])

def scaled_linear_func(x):
    return torch.sum(x,5)

