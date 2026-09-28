from time import sleep
import pandas as pd
from tqdm import tqdm
import json
import math

import requests

# Replace with your actual username from GeoNames
GEONAMES_USERNAME = "climatepub4kg"

def convert_positive_east_to_wgs84(bbox):
    """
    Converts a positive east (0 to 360 degrees) bounding box to WGS84(-180 to +180 degrees) format.
    The input bbox should be a dictionary with keys: 'north', 'south', 'east', 'west'.
    The output will be a dictionary with the same keys, but in WGS84 format.
    """
    return {
        'north': bbox['north'],
        'south': bbox['south'],
        'east': bbox['east']- 180.0,
        'west': bbox['west']- 180.0
    }

def do_bounding_boxes_intersect(inner_box, outer_box):
    """
    Returns True if box1 and box2 intersect.
    Each box is a dict with keys: 'north', 'south', 'east', 'west'.
    Both boxes should be in the same coordinate system. i.e. north and south are -90 to 90 and east and west are -180 to 180.
    """
    return (
        outer_box['south'] <= inner_box['south'] and
        inner_box['north'] <= outer_box['north'] and
        outer_box['west'] <= inner_box['west'] and
        inner_box['east'] <= outer_box['east']
    )

def is_coordinate_in_bounding_box(lat, lng, box):
    return (
        box['south'] <= lat <= box['north'] and
        box['west'] <= lng <= box['east']
    )

def get_intersecting_geonameids(bbox,geonameid_lookup=None,continents_df=None):
    """
    Given a bounding box, return a list of geonameids that intersect with it.
    The input bbox should be a dictionary with keys: 'north', 'south', 'east', 'west'.
    """
    # Load the geonameid lookup file
    if geonameid_lookup is None:
        with open("geolocation/geonameid_lookup.json", 'r') as f:
            geonameid_lookup = json.load(f)

    if continents_df is None:
        continents_df= pd.read_csv("geolocation/continents.csv",keep_default_na=False)
    # countries_df= pd.read_csv("geolocation/countries.csv")

    intersecting_ids = set()
    # converted_bbox=convert_positive_east_to_wgs84(bbox)
    converted_bbox=bbox
    for geonameid, data in geonameid_lookup.items():
        if 'north' in data and 'south' in data and 'east' in data and 'west' in data and data['north'] is not None and data['south'] is not None and data['east'] is not None and data['west'] is not None:
            data_bbox = {
            'north': data['north'],
            'south': data['south'],
            'east': data['east'],
            'west': data['west']
            }
            if do_bounding_boxes_intersect(data_bbox, converted_bbox):
                intersecting_ids.add(geonameid)
                # Check if data contains country key, if yes, add the continent geonameid of that country
                if 'country' in data and 'continent' in data and data['continent'] is not None:
                    continent_geonameid=continents_df.loc[continents_df['iso'] == data['continent'], 'geonameid'] 
                    if len(continent_geonameid) > 0:
                        intersecting_ids.add(list(continent_geonameid.values)[0])
                    
        else:
            if 'latitude' in data and 'longitude' in data:
                lat = data['latitude']
                lng = data['longitude']
                # Check if the coordinate is within the bounding box
                if is_coordinate_in_bounding_box(lat, lng, converted_bbox):
                    intersecting_ids.add(geonameid)

    return list(intersecting_ids)

def add_ocean_metadata():
    df=pd.read_csv("geolocation/water.csv")
    all_genome_ids=df.iloc[:, 0]
    df['north'] = None
    df['south'] = None
    df['east'] = None
    df['west'] = None
    for each_id in tqdm(all_genome_ids):
        url = f"http://api.geonames.org/getJSON?geonameId={each_id}&username={GEONAMES_USERNAME}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'bbox' in data:
                if 'north' in data['bbox'] and 'south' in data['bbox'] and 'east' in data['bbox'] and 'west' in data['bbox']:
                    north = data['bbox']['north']
                    south = data['bbox']['south']
                    east = data['bbox']['east']
                    west = data['bbox']['west']
                    df.loc[df.iloc[:, 0] == each_id, 'north'] = north
                    df.loc[df.iloc[:, 0] == each_id, 'south'] = south
                    df.loc[df.iloc[:, 0] == each_id, 'east'] = east
                    df.loc[df.iloc[:, 0] == each_id, 'west'] = west
        else:
            print(f"Error fetching data for ID {each_id}: {response.status_code}")
        sleep(1)
    df.to_csv("geolocation/water.csv", index=False)
    print("Added metadata information in water.")

def add_country_metadata():
    df=pd.read_csv("geolocation/countries.csv")
    all_genome_ids=list(df.loc[:,'geonameid'].values)
    df['north'] = None
    df['south'] = None
    df['east'] = None
    df['west'] = None
    for each_id in tqdm(all_genome_ids):
        url = f"http://api.geonames.org/getJSON?geonameId={each_id}&username={GEONAMES_USERNAME}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'bbox' in data:
                if 'north' in data['bbox'] and 'south' in data['bbox'] and 'east' in data['bbox'] and 'west' in data['bbox']:
                    north = data['bbox']['north']
                    south = data['bbox']['south']
                    east = data['bbox']['east']
                    west = data['bbox']['west']
                    df.loc[df.loc[:, 'geonameid'] == each_id, 'north'] = north
                    df.loc[df.loc[:, 'geonameid'] == each_id, 'south'] = south
                    df.loc[df.loc[:, 'geonameid'] == each_id, 'east'] = east
                    df.loc[df.loc[:, 'geonameid'] == each_id, 'west'] = west
        else:
            print(f"Error fetching data for ID {each_id}: {response.status_code}")
        sleep(1)
    df.to_csv("geolocation/countries.csv", index=False)
    print("Added metadata information in countries.")

def add_continent_metadata():
    df=pd.read_csv("geolocation/continents.csv")
    all_genome_ids=list(df.loc[:,'geonameid'].values)
    df['north'] = None
    df['south'] = None
    df['east'] = None
    df['west'] = None
    for each_id in tqdm(all_genome_ids):
        url = f"http://api.geonames.org/getJSON?geonameId={each_id}&username={GEONAMES_USERNAME}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'bbox' in data:
                if 'north' in data['bbox'] and 'south' in data['bbox'] and 'east' in data['bbox'] and 'west' in data['bbox']:
                    north = data['bbox']['north']
                    south = data['bbox']['south']
                    east = data['bbox']['east']
                    west = data['bbox']['west']
                    df.loc[df.loc[:, 'geonameid'] == each_id, 'north'] = north
                    df.loc[df.loc[:, 'geonameid'] == each_id, 'south'] = south
                    df.loc[df.loc[:, 'geonameid'] == each_id, 'east'] = east
                    df.loc[df.loc[:, 'geonameid'] == each_id, 'west'] = west
        else:
            print(f"Error fetching data for ID {each_id}: {response.status_code}")
        sleep(1)
    df.to_csv("geolocation/continents.csv", index=False)
    print("Added metadata information in continents.")

def add_country_subdivision_metadata():
    csd_df=pd.read_csv("geolocation/country_subdivision.csv")
    all_genome_ids=list(csd_df.loc[:,'geonameid'].values)
    for each_id in tqdm(all_genome_ids):
        # Check if the geonameid already exists in the saved dataframe
        if not math.isnan(csd_df.loc[csd_df.loc[:,'geonameid']==each_id,'north'].values[0]):
            # If it exists, skip to the next iteration
            continue
        url = f"http://api.geonames.org/getJSON?geonameId={each_id}&username={GEONAMES_USERNAME}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'bbox' in data:
                if 'north' in data['bbox'] and 'south' in data['bbox'] and 'east' in data['bbox'] and 'west' in data['bbox']:
                    north = data['bbox']['north']
                    south = data['bbox']['south']
                    east = data['bbox']['east']
                    west = data['bbox']['west']
                    csd_df.loc[csd_df.loc[:, 'geonameid'] == each_id, 'north'] = north
                    csd_df.loc[csd_df.loc[:, 'geonameid'] == each_id, 'south'] = south
                    csd_df.loc[csd_df.loc[:, 'geonameid'] == each_id, 'east'] = east
                    csd_df.loc[csd_df.loc[:, 'geonameid'] == each_id, 'west'] = west
        else:
            print(f"Error fetching data for ID {each_id}: {response.status_code}")
        sleep(0.2)
    csd_df.to_csv("geolocation/country_subdivision.csv", index=False)

def cities_to_csv():

    # Define column names based on GeoNames schema
    columns = [
        "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
        "feature_class", "feature_code", "country_code", "cc2",
        "admin1_code", "admin2_code", "admin3_code", "admin4_code",
        "population", "elevation", "dem", "timezone", "modification_date"
    ]

    # Load the tab-separated file
    df = pd.read_csv("geolocation/cities15000.txt", sep="\t", header=None, names=columns, dtype=str)

    # Optional: Convert some columns to appropriate types (e.g., numeric)
    df["geonameid"] = pd.to_numeric(df["geonameid"], errors='coerce')
    df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
    df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce')
    df["population"] = pd.to_numeric(df["population"], errors='coerce')
    df["elevation"] = pd.to_numeric(df["elevation"], errors='coerce')
    df["dem"] = pd.to_numeric(df["dem"], errors='coerce')

    # Save as CSV
    df.to_csv("geolocation/cities.csv", index=False)
    print("CSV file for cities created successfully.")

def countries_to_csv():

    # Define column names based on GeoNames schema
    columns = [
        "iso",
        "iso3",
        "isonumeric",
        "fips",
        "country",
        "capital",
        "area_sqkm",
        "population",
        "continent",
        "tld",
        "currencycode",
        "currencyname",
        "phone",
        "postal_code_format",
        "postal_code_regex",
        "languages",
        "geonameid",
        "neighbours",
        "equivalent_fips_code"
    ]

    # Load the tab-separated file
    df = pd.read_csv("geolocation/countries.txt", sep="\t", header=None, names=columns, dtype=str)

    # Optional: Convert some columns to appropriate types (e.g., numeric)
    df["geonameid"] = pd.to_numeric(df["geonameid"], errors='coerce')
    df["population"] = pd.to_numeric(df["population"], errors='coerce')

    # Save as CSV
    df.to_csv("geolocation/countries.csv", index=False)
    print("CSV file for countries created successfully.")

def no_country_to_csv():

    # Define column names based on GeoNames schema
    # Define column names based on GeoNames schema
    columns = [
        "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
        "feature_class", "feature_code", "country_code", "cc2",
        "admin1_code", "admin2_code", "admin3_code", "admin4_code",
        "population", "elevation", "dem", "timezone", "modification_date"
    ]

    # Load the tab-separated file
    df = pd.read_csv("geolocation/no-country.txt", sep="\t", header=None, names=columns, dtype=str)

    # Optional: Convert some columns to appropriate types (e.g., numeric)
    df["geonameid"] = pd.to_numeric(df["geonameid"], errors='coerce')
    df["latitude"] = pd.to_numeric(df["latitude"], errors='coerce')
    df["longitude"] = pd.to_numeric(df["longitude"], errors='coerce')
    df["population"] = pd.to_numeric(df["population"], errors='coerce')
    df["elevation"] = pd.to_numeric(df["elevation"], errors='coerce')
    df["dem"] = pd.to_numeric(df["dem"], errors='coerce')

    # Save as CSV
    df.to_csv("geolocation/no-country.csv", index=False)
    print("CSV file for no country created successfully.")

def create_geonameid_lookup():
    continents_df= pd.read_csv("geolocation/continents.csv",keep_default_na=False)
    countries_df= pd.read_csv("geolocation/countries.csv",keep_default_na=False)
    cities_df= pd.read_csv("geolocation/cities.csv")
    csd_df=pd.read_csv("geolocation/country_subdivision.csv")
    
    
    # no_country_df= pd.read_csv("geolocation/no-country.csv")
    # water_df= pd.read_csv("geolocation/water.csv")
    # No country and water body merged
    no_country_df= pd.read_csv("geolocation/no-country-water-merged.csv")
    
    file_path= "geolocation/geonameid_lookup.json"


    continents_dict=continents_df.set_index('geonameid').to_dict(orient='index')
    countries_dict=countries_df.set_index('geonameid').to_dict(orient='index')
    # water_dict=water_df.set_index('geonameid').to_dict(orient='index')
    cities_dict=cities_df.set_index('geonameid').to_dict(orient='index')
    no_country_dict=no_country_df.set_index('geonameid').to_dict(orient='index')
    csd_dict=csd_df.set_index('geonameid').to_dict(orient='index')
    all_dicts = {**continents_dict, **countries_dict, **cities_dict, **no_country_dict, **csd_dict}
    with open(file_path, 'w') as f:
        json.dump(all_dicts, f,indent=4)
    
    print(f"Geonameid lookup file created at {file_path}")

def create_domain_to_geonameid_lookup():
    """
    Create a lookup table that maps domain names to geonameids.
    """
    df=pd.read_csv("geolocation/cordex_domain_wgs4_coordinates.csv")
    domain_to_geoname_ids={}
    for _, row in df.iterrows():
        domain = row['CORDEX_domain']
        bbox={
            'north':row['north'], 
            'south':row['south'], 
            'east': row['east'], 
            'west': row['west']
        }
        intersecting_ids = get_intersecting_geonameids(bbox)
        if intersecting_ids:
            domain_to_geoname_ids[domain] = [str(id) for id in intersecting_ids]
            
    # Save the lookup table to a JSON file
    with open("geolocation/domain_to_geonameid_lookup.json", 'w') as f:
        json.dump(domain_to_geoname_ids, f)

def merge_water_bodies_and_no_country_region():
    # Load the water bodies and no country region data
    water_df=pd.read_csv("geolocation/water.csv")
    no_country_df=pd.read_csv("geolocation/no-country.csv")

    # Rename the 'Name' column in water_df to 'name' for consistency
    water_df=water_df.rename(columns={"Name":"name"})
    
    # Combine the two dataframes
    combined_df = pd.concat([water_df, no_country_df], ignore_index=True)

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv("geolocation/no-country-water-merged.csv", index=False)
    print("Merged water bodies and no country region into water_and_no_country.csv")


if __name__ == "__main__":
    # cities_to_csv()
    # countries_to_csv()
    # no_country_to_csv()
    # add_ocean_metadata()
    # add_country_metadata()
    # add_continent_metadata()
    # create_geonameid_lookup()
    # add_country_subdivision_metadata()
    
    create_domain_to_geonameid_lookup()

    # demo_bbox={
    #     'north': 12.21, # max lat
    #     'south': -52.57, # min lat
    #     'east':  180, # max long
    #     'west':  -180 # min long
    # }
    # intersecting_ids = get_intersecting_geonameids(demo_bbox)
    # print(intersecting_ids)
    # # open reverse geonameid lookup file
    # with open("geolocation/geonameid_lookup.json", 'r') as f:
    #     geonameid_lookup = json.load(f)
    # # print the intersecting names
    # for geonameid in intersecting_ids:
    #     if geonameid in geonameid_lookup:
    #         print(f"Geoname ID: {geonameid}, Name: {geonameid_lookup[geonameid].get('name','N/A')}")
    #     else:
    #         print(f"Geoname ID: {geonameid} not found in lookup.")
