from dataclasses import dataclass
from dataclasses_json import dataclass_json
import json
import sys
import csv
from utilities import *
from tqdm import tqdm
import numpy as np

with open('proc_tracks.json', 'r+') as pt:
    cyclone_dict = json.load(pt)

@dataclass_json
@dataclass
class Track:
    """
    This class contains relevant data for a given cyclone track.
    It should be working as intended with one being able to create the track from an SID and the tracks JSON.
    For example:
    >>> track = Track.from_sid('1992096S08132')
    >>> coordinates = track.coordinates
    """
    iso_times: list[str]
    categories: list[int]
    coordinates: list[tuple[float,float]]
    wind_speeds: list[float]
    pressures: list[float]
    basin: list[str]
    subbasin: list[str]
    data_len: int
    season: int
    sid: str = ""
    
    # We define a raw track as the rows from the CSV file with the same SID
    def import_from_raw(self,raw_track,storm_id):
        self.data_len = len(raw_track)
        self.season = raw_track[0]['SEASON']

        for t_row in raw_track:
            self.iso_times.append(t_row['ISO_TIME'])
            self.categories.append(t_row['USA_SSHS'])
            self.coordinates.append((t_row['LON'], t_row['LAT']))
            self.wind_speeds.append(t_row['WMO_WIND'])
            self.pressures.append(t_row['WMO_PRES'])
            self.basin.append(t_row['BASIN'])
            self.subbasin.append(t_row['SUBBASIN'])
    
    def save(self, json_file):
        add_to_json(self.sid, self.data, json_file)

    @classmethod
    def from_sid(cls, sid):
        self = cls.from_dict(cyclone_dict[sid])
        self.sid = sid
        return self

def save_all_storms(file_name, save_to_file='tracks.json', year_init=1979):
    with open(file=file_name, newline='') as storm_file:
        current_raw_track = []
        current_storm = ''

        # See first row for heading names
        reader = csv.DictReader(storm_file)

        current_raw_track = []

        for i, row in tqdm(enumerate(reader)):
            if i < 2:
                continue

            if row['TRACK_TYPE'] != 'main':
                continue

            if (int(row['SEASON']) >= year_init):
                if current_raw_track == []:
                    current_raw_track = [row]
                    current_storm = row['SID']
                    continue

                if (current_storm != row['SID']):
                    track = Track()
                    track.import_from_raw(current_raw_track, current_storm)
                    track.save(save_to_file)
                    current_raw_track = [row]
                    current_storm = row['SID']
                else:
                    current_raw_track.append(row)
            else:
                continue

def convert_lat_lon(tracks_file='tracks.json', proc_tracks_file='proc_tracks.json'):
    with open('tracks.json', 'r') as tracks_json:
        tracks_dict = json.load(tracks_json)
    
    for sid, data in tqdm(tracks_dict.items()):
        new_dict = data
        new_coords = []
        for coordinate in data['coordinates']:
            new_longitude = (float(coordinate[0]) % 360 + 540) % 360 - 180
            new_coords.append((new_longitude, float(coordinate[1])))
        new_dict['coordinates'] = new_coords
        add_to_json(sid, new_dict, proc_tracks_file)

def all_avaliable_tracks(tracks_file='proc_tracks.json', data_local='/g/data/x77/ob2720/cyclone_binaries'):
    with open(tracks_file, 'r') as tracks_json:
        tracks_dict = json.load(tracks_json)
    
    cyclones_saved = os.listdir(data_local)
    
    print(cyclones_saved)
    
    for sid in cyclones_saved:
        if sid[:-3] in tracks_dict:
            previous_month = np.datetime64(tracks_dict[sid[:-3]]['iso_times'][0]).astype(object).month
            for iso_time in tracks_dict[sid[:-3]]['iso_times']:
                current_month = (np.datetime64(iso_time)).astype(object).month
                if current_month != previous_month:
                    break
                previous_month = (np.datetime64(iso_time)).astype(object).month
            else:
                data = {sid:tracks_dict[sid[:-3]]}
                append_to_json('available.json', data)
    
    

if __name__ == '__main__':
    # save_all_storms('ibtracs.ALL.list.v04r00.csv', save_to_file="tracks.json", year_init=1979)
    all_avaliable_tracks()