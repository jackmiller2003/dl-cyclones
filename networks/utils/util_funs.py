import math
import torch

def get_cyclone_distance_error(output_tensor:torch.Tensor, target_tensor: torch.Tensor):

    pred_location = target_tensor[:,0:2,0] + output_tensor[:,0:2]
    true_location = target_tensor[:,0:2,1]

    R = 6371e3

    lon0, lat0 = true_location[0], true_location[1]
    lon1,lat1 = pred_location[0], pred_location[1]

    phi0 = lat0 * (math.pi/180) # Rads
    phi1 = lat1 * (math.pi/180)

    delta_phi = phi1 - phi0
    delta_lambda = (lon1 - lon0) * (math.pi/180)

    a = torch.pow(torch.sin(delta_phi/2),2) + torch.cos(phi0) * torch.cos(phi1) * torch.pow(torch.sin(delta_lambda/2),2) 
    a = a.float()
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    c = c.double()
    c = R * c

    mse = torch.sum(c)/c.shape[0]
    
    return mse


