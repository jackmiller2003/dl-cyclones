from utils.loss_functions import L2_Dist_Func_Intensity
from utils.data_loader import *
import torch
from datetime import datetime
from models.uv_model import UV_Model
from models.z_model import Z_Model
from models.meta_model import Meta_Model
from models.fusion_model import Fusion_Model
import os
import datetime
import xarray
import matplotlib.pyplot as plt
from utils.util_funcs import *
from tqdm import tqdm
from collections import OrderedDict
import re
import pickle
import json
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
import geopandas as gpd
from geopandas import GeoDataFrame
import geoplot
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt

data_dir = '/g/data/x77/jm0124/test_holdout/'
models_dir = '/g/data/x77/jm0124/models'

test_path = '/home/156/jm0124/dl-cyclones/tracks/test.json'

with open(test_path, 'r') as test_json:
    test_dict = json.load(test_json)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in miles. Use 6371 for kilometers
    return c * r

def eval_on_cyclone(cyclone_id, model):
    model.eval()

    examples, labels = get_examples_and_labels(cyclone_id)
    pred_points = []
    true_points = []
    original_points = []
    distances = []

    for i in range(0, len(examples)):
        pred = model.forward(examples[i]).detach().numpy()

        label = labels[i].detach().numpy()

        original_point = Point(float(label[0][0]), float(label[1][0]))
        true_point = Point(float(label[0][1]), float(label[1][1]))
        pred_point = Point(
            float(label[0,0]) + pred[0,0],
            float(label[1,0]) + pred[0,1]
        )

        original_points.append((float(label[0][0]), float(label[1][0])))
        true_points.append((float(label[0][1]), float(label[1][1])))
        pred_points.append((
            float(label[0,0]) + pred[0,0],
            float(label[1,0]) + pred[0,1]
        ))

        distance = haversine(float(label[0][1]), float(label[1][1]), float(label[0,0]) + pred[0,0], float(label[1,0]) + pred[0,1])
        distances.append(distance)

        # print(f"Original point {original_point}")
        # print(f"True point {true_point}")
        # print(f"Predicted point {pred_point}")
        # print(f"Distance is {distance}")

        """
        tensor([[-35.1598, -40.1572],
        [ 12.6604,  13.5424],
        [ -1.0000,  -1.0000]], dtype=torch.float64)

        tensor([[-5.3318e+00,  6.7186e-01, -1.1475e-13]], grad_fn=<AddmmBackward0>)
        """

    data_frame_dict = {}
    data_frame_dict['colour'] = []
    data_frame_dict['lon'] = []
    data_frame_dict['lat'] = []
    
    for (lon, lat) in true_points:
        data_frame_dict['colour'].append('blue')
        data_frame_dict['lon'].append(lon)
        data_frame_dict['lat'].append(lat)
    
    for (lon, lat) in pred_points:
        data_frame_dict['colour'].append('red')
        data_frame_dict['lon'].append(lon)
        data_frame_dict['lat'].append(lat)

    true_polygons = []
    pred_polygons = []

    polygons_dict = {}
    polygons_dict['colour'] = []
    polygons_dict['geometry'] = []

    for i in range(0,len(original_points)):
        true_polygons.append(LineString([(true_points[i][0], true_points[i][1]), (original_points[i][0], original_points[i][1])]))
        pred_polygons.append(LineString([(original_points[i][0], original_points[i][1]), (pred_points[i][0], pred_points[i][1])]))

    for true_polygon in true_polygons:
        polygons_dict['colour'].append('blue')
        polygons_dict['geometry'].append(true_polygon)
    
    for pred_polygon in pred_polygons:
        polygons_dict['colour'].append('red')
        polygons_dict['geometry'].append(pred_polygon)
    
    for (lon, lat) in original_points:
        polygons_dict['colour'].append('black')
        polygons_dict['geometry'].append(Point(lon,lat))

    print(polygons_dict)

    df = pd.DataFrame(data_frame_dict)
    df2 = pd.DataFrame(polygons_dict)

    gdf = GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    gdf2 = GeoDataFrame(df2, geometry=df2['geometry'])

    print(gdf2)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    base = world.plot(color='white', edgecolor='black')

    minx, miny, maxx, maxy = gdf2.total_bounds
    base.set_xlim(minx-5, maxx+5)
    base.set_ylim(miny-5, maxy+5)

    gdf2.plot(ax=base, color=gdf2['colour'], alpha=0.9)

    square_distance = 0

    for distance in distances:
        square_distance += distance**2
    
    rmse = (square_distance/len(distances))**(1/2)

    print(f"{cyclone_id} and {rmse}")

    plt.savefig(f'world-{cyclone_id}-{rmse}.jpg', dpi=800)
    

def get_examples_and_labels(cyclone):
    time_step_back = 1

    j = 2
    bound = 9

    examples = []
    labels = []

    target_parameters = [0,1]

    data = test_dict[cyclone]

    for coordinate in data['coordinates'][:-bound]:
        cyclone_ds = xarray.open_dataset(data_dir+cyclone)
        cyclone_ds_new = cyclone_ds[dict(time=list(range(j-time_step_back-1,j)))]
        
        if target_parameters == [0,1]:
            cyclone_ds_new = cyclone_ds_new[['u','v']]
        elif target_parameters == [2]:
            cyclone_ds_new = cyclone_ds_new[['z']]
        
        cyclone_ds_crop_new = cyclone_ds_new.to_array().to_numpy()
        cyclone_ds_crop_new = np.transpose(cyclone_ds_crop_new, (1, 0, 2, 3, 4))
        
        example = torch.from_numpy(cyclone_ds_crop_new)
        num_channels = int(5*len(target_parameters)*(1+time_step_back))

        example = torch.reshape(example, (1,num_channels,160,160))

        label = torch.from_numpy(np.array([[
                                    float(data['coordinates'][j-1][0]), float(data['coordinates'][j+bound-2][0])], 
                                    [float(data['coordinates'][j-1][1]), float(data['coordinates'][j+bound-2][1])],
                                    [float(data['categories'][j-1]), float(data['categories'][j])]
                                            ]))
        
        if torch.isnan(example).any():
            print(f"Example size is: {example.size()}")
            print(f"Cyclone: {cyclone}")
            print(f"Coordinate: {coordinate}")
        
        examples.append(example)
        labels.append(label)

        j += 1

    return examples, labels

if __name__ == "__main__":

    model_uv = UV_Model()

    state = torch.load(f'{models_dir}/model_uv-64.7119321766501')            
    state_dict = state['state_dict']
    
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    model_uv.load_state_dict(model_dict)

    eval_on_cyclone('1986225N17134.nc', model_uv)