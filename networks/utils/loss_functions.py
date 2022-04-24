import torch
import math
import numpy as np

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
        c = 2 * R * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
        c = c.double()
        
        # i = intensity_scale * (intensity_func(true_intensity) * (true_intensity-pred_intensity)) + 1e-6
        
        mean_dist_loss = torch.sqrt(torch.sum(torch.pow(c,2))/output_tensor.shape[0])
        
        loss_out = mean_dist_loss

        return loss_out

def test_loss():
    sydney = (-33.8688, 151.2093)
    canberra = (-35.2802, 149.1310)

    target_tensor = np.array([[
        [0, 151.2093],
        [0, -33.8688],
        [0,0]
                                ],
        [
        [0, 153.0260],
        [0, -27.4705],
        [0,0]
                                ]])
    
    pred_tensor = np.array([
        [149.1310, -35.2802, 0],
        [145.7710, -16.9203, 0]
                            ])

    loss_fn = L2_Dist_Func_Intensity()

    loss = loss_fn(torch.from_numpy(pred_tensor), torch.from_numpy(target_tensor), 0)

    print(loss)

if __name__ == "__main__":
    test_loss()