import json
import sys
import csv
from utilities import *
from tqdm import tqdm

class Track:
    """
    This class contains relevant data for a given cyclone track.
    """

    def __init__(self, sid):

        self.data = {
            'iso_times':[],
            'categories':[],
            'coordinates':[],
            'wind_speeds':[],
            'pressures':[],
            'storm_id':0,
            'data_len':0,
            'season':0
        }

        self.sid = sid
    
    # We define a raw track as the rows from the CSV file with the same SID
    def import_from_raw(self,raw_track,storm_id):
        self.data['storm_id'] = storm_id
        self.data['data_len'] = len(raw_track)
        self.data['season'] = raw_track[0]['SEASON']

        for t_row in raw_track:
            self.data['iso_times'].append(t_row['ISO_TIME'])
            self.data['categories'].append(t_row['USA_SSHS'])
            self.data['coordinates'].append((t_row['LAT'], t_row['LON']))
            self.data['wind_speeds'].append(t_row['WMO_WIND'])
            self.data['pressures'].append(t_row['WMO_PRES'])
    
    def save(self, json_file):
        add_to_json(self.sid, self.data, json_file)

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

            if (int(row['SEASON']) > year_init):
                if current_raw_track == []:
                    current_raw_track = [row]
                    current_storm = row['SID']
                    continue

                if (current_storm != row['SID']):
                    track = Track(current_storm)
                    track.import_from_raw(current_raw_track, current_storm)
                    track.save(save_to_file)
                    current_raw_track = [row]
                    current_storm = row['SID']
                else:
                    current_raw_track.append(row)

            else:
                continue

if __name__ == '__main__':
    save_all_storms('ibtracs.ALL.list.v04r00.csv', save_to_file="tracks.json", year_init=1979)