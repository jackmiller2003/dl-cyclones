# import json
# import xarray

# def get_ds(cyclone_json, config_dict):
#     """
#     Returns a list of data selections which surround the cyclones.
#     """

#     atm_params = config_dict['atm_params']
#     cyclone_selections = []

#     # Here relevant selections is a list of lists (param x year data selections)
#     relevant_selections = []

#     for param in atm_params:
#         year_list = []
#         for year in range(1979,2021):
#             current_ds = xarray.open_mfdataset(f'/g/data/rt52/era5/pressure-levels/reanalysis/{param}/{year}/{param}_era5_oper_pl_*.nc', \
#                 combine='nested', concat_dim='time')
#             year_list.append(current_ds)

#         relevant_selections.append(year_list)
#     try:
#         with open(cyclone_json, 'r') as f:
#             cyclone_dictionary = json.load(f)
#     except:
#         print("Unable to open track JSON.")


#     for label, data in cyclone_dictionary.items():
#         coordinates = data['coordinates']
#         times = data['iso_times']

#         # Needed check as otherwise get_track_ds won't run properly
#         if len(coordinates) == len(times):
#             raise Exception('Coordinates not the same size as times')
        
#         cyclone_selections.append(get_track_ds(coordinates,times,relevant_selections))

# def get_track_ds(coordinates, times, relevant_selections):
#     """
#     Returns a list of data selections around a single cyclone.
#     """

#     for coordinate in coordinates:
#         for time in times:
#             # Need to worry about whether a cyclone goes over two years :(
            
#             if time[0].year != time[-1].year:
                







    


