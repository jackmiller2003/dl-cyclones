import torch
import math

def scaled_linear_func(x):
    return x+5

class L2_Dist_Func_Intensity(torch.nn.Module):
    def __init__(self):
        super(L2_Dist_Func_Intensity, self).__init__()
    
    def forward(self, output_tensor:torch.Tensor, true_tensor: torch.Tensor, intensity_func=scaled_linear_func, intensity_scale=1e-2):
        """
        Here we have two inputs:
            * Predicted location -> (lon, lat, intensity)
            * True location -> (lon, lat, intensity)
        """
        
        # print(f"Output tensor: {output_tensor.size()}")
        # print(f"True tensor: {true_tensor.size()}")

        pred_location = output_tensor[:,0:2]
        true_location = true_tensor[:, 0:2]
        pred_intensity = output_tensor[:,2]
        pred_intensity = pred_intensity.view(-1,1)
        true_intensity = true_tensor[:,2]
        true_intensity = true_intensity.view(-1,1)

        # print(f"Predicted location: {pred_location}")
        # print(f"True location: {true_location}")
        # print(f"Predicted intensity: {pred_intensity}")
        # print(f"True intensity: {true_intensity}")
        
        R = 6371e3

        lon0, lat0 = true_location[:,0], true_location[:,1]
        lon1,lat1 = pred_location[:,0], pred_location[:,1]
        
        # print(f"lat0 = {lat0}")
        # print(f"lat1 = {lat1}")

        phi0 = lat0 * (math.pi/180) # Rads
        phi1 = lat1 * (math.pi/180)
        
        # print(phi0)
        # print(phi1)

        delta_phi = phi1 - phi0
        delta_lambda = (lon1 - lon0) * (math.pi/180)

        a = torch.pow(torch.sin(delta_phi/2),2) + torch.cos(phi0) * torch.cos(phi1) * torch.pow(torch.sin(delta_lambda/2),2)
        a = a.float()
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        c = c.double()
        # cyclone_dist = R * c
        
        i = intensity_func(true_intensity) * (true_intensity-pred_intensity)
        i = torch.pow(i,2)
        
        mean_dist_loss = torch.sum(c)/c.shape[0]
        mean_intensity_loss = torch.sum(i)/i.shape[0]
        
        loss_out = mean_dist_loss + mean_intensity_loss
        
        print(f"Loss is {loss_out}")
        print(f"c is {c}")

        return loss_out



