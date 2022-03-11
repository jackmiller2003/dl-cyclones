import torch
import math

def scaled_linear_func(x):
    return torch.sum(x,5)

class L2_Dist_Func_Intensity(torch.nn.Module):
    def __init__(self):
        super(L2_Loss, self).__init__()
    
    def forward(self, output_tensor:torch.Tensor, true_tensor: torch.Tensor, intensity_func=scaled_linear_func, intensity_scale=1e-2):
        """
        Here we have two inputs:
            * Predicted location -> (lon, lat, intensity)
            * True location -> (lon, lat, intensity)
        """

        pred_location = output_tensor[0:2]
        true_location = true_tensor[0:2]
        pred_intensity = output_tensor[2]
        true_intensity = true_tensor[2]

        R = 6371e3

        lon0, lat0 = true_location[0], true_location[1]
        lon1,lat1 = pred_location[0], pred_location[1]

        phi0 = lat0 * (math.pi/180) # Rads
        phi1 = lat1 * (math.pi/180)

        delta_phi = phi1 - phi0
        delta_lambda = (lon1 - lon0) * (math.pi/180)

        a = torch.pow(torch.sin(delta_phi/2),2) + torch.cos(phi0) + torch.cos(phi1) + torch.pow(torch.sin(delta_lambda/2),2)
        a = a.float()
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        c = c.double()
        # cyclone_dist = R * c

        return torch.sum(torch.pow(c, 2), intensity_scale * intensity_func(pred_intensity - true_intensity) * torch.pow((pred_intensity - true_intensity), 2))



