import math
import torch

def get_cyclone_distance_error(output_tensor:torch.Tensor, target_tensor: torch.Tensor):

    pred_location = target_tensor[:,0:2,0] + output_tensor[:,0:2]
    true_location = target_tensor[:,0:2,1]
    
    print(f"Prev local: {target_tensor[:,0:2,0][0]}")
    print(f"Estimated vector: {output_tensor[:,0:2][0]}")
    print(f"Pred lcoal: {(pred_location[0])}")
    print(f"True local: {target_tensor[:,0:2,1][0]}")

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

def calculate_average_mse(model, validation_loader_uvzmeta):
    model.train(False)

    running_mse = 0
    mse_graph = []

    for i, return_list in enumerate(validation_loader_uvzmeta):

        atm_data = return_list[0][0]
        meta_data = return_list[1][0]

        target = return_list[0][1]

        output = model(atm_data, meta_data) # God I hope this works lol

        error = get_cyclone_distance_error(output, target)

        running_mse += error

        if i % 10 == 9:
            last_mse = running_mse / 10 # loss per batch
            mse_graph.append(last_mse)
            print('batch {} mse: {}'.format(i + 1, last_mse))
            running_mse = 0
        
        if i == 101:
            break
    
    return mse_graph
