import torch
import math

def scaled_linear_func(x):
    return x+5

class L2_Dist_Func_Intensity(torch.nn.Module):
    def __init__(self):
        super(L2_Dist_Func_Intensity, self).__init__()
    
    def forward(self, output_tensor:torch.Tensor, target_tensor: torch.Tensor, rank, intensity_func=scaled_linear_func, intensity_scale=1e-2):
        """
        Here we have two inputs:
            * Predicted location -> (lon disp., lat disp., change in intensity)
            * Target location -> (lon (t-1, t), lat (t-1, t), intensity (t-1,t))
        """
        
        # output_tensor = output_tensor.to(rank)
        # target_tensor = target_tensor.to(rank)

        # print(f"Target tensor: {target_tensor.size()}")
        # print(f"Output tensor: {output_tensor.size()}")

        pred_location = target_tensor[:,0:2,0] + output_tensor[:,0:2]
        true_location = target_tensor[:,0:2,1]
        # pred_intensity = target_tensor[:,2,0] + output_tensor[:,2]
        # pred_intensity = pred_intensity.view(-1,1)
        # true_intensity = target_tensor[:,2,1]
        # true_intensity = true_intensity.view(-1,1)

        # print(f"Predicted location: {pred_location[0]}")
        # print(f"True location: {true_location[0]}")
        # print(f"Predicted intensity: {pred_intensity[0]}")
        # print(f"True intensity: {true_intensity[0]}")
        
        # R = 6371e3
        R = 6371 # in km

        lon0, lat0 = true_location[:,0], true_location[:,1]
        lon1,lat1 = pred_location[:,0], pred_location[:,1]

        phi0 = lat0 * (math.pi/180) # Rads
        phi1 = lat1 * (math.pi/180)

        delta_phi = phi1 - phi0
        delta_lambda = (lon1 - lon0) * (math.pi/180)

        a = torch.pow(torch.sin(delta_phi/2),2) + torch.cos(phi0) * torch.cos(phi1) * torch.pow(torch.sin(delta_lambda/2),2) 
        a = a.float()
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        c = c.double()
        c = R * c
        
        # i = intensity_scale * (intensity_func(true_intensity) * (true_intensity-pred_intensity)) + 1e-6

        # print(f"c {c}")
        # print(f"i {i}")
        
        mean_dist_loss = torch.sqrt(torch.sum(torch.pow(c,2))/output_tensor.shape[0])
        # mean_intensity_loss = torch.sum(i)/i.shape[0]
        
        loss_out = mean_dist_loss #+ mean_intensity_loss
        
        # loss_out = loss_out.to(rank)
        
        # print(f"Loss is {loss_out}")
        # print(f"c is {c}")

        return loss_out



