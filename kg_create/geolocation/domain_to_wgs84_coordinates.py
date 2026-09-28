import cordex as cx
import pandas as pd
from tqdm import tqdm

def find_bbox_coordinates(domain):
    # 1. Load a CORDEX domain dataset (e.g. EUR-11, EUR-44, SAM-44 etc.)
    ds = cx.domain(domain)

    # 2. Use the transform_bounds accessor to compute true lon/lat bounds
    #    Outputs: min_lon, min_lat, max_lon, max_lat in geographic CRS (EPSG:4326)

    # If the domain does not have pollon and pollat, we need to handle it differently
    if domain[-1] == 'i':
        min_lon = ds['lon'].min().item()
        max_lon = ds['lon'].max().item()
        min_lat = ds['lat'].min().item()
        max_lat = ds['lat'].max().item()
        return max_lat, min_lat, max_lon, min_lon
    
    bounds = cx.transform_bounds(ds)
    min_lon = bounds['lon'].min().item()
    max_lon = bounds['lon'].max().item()
    min_lat = bounds['lat'].min().item()
    max_lat = bounds['lat'].max().item()
    # return north, south, east, west
    return max_lat, min_lat, max_lon, min_lon


df=pd.read_csv("rotated-latitude-longitude.csv")
domains=df["CORDEX_domain"].unique()
north=[]
south=[]
east=[]
west=[]
for each_domain in tqdm(domains):
    n,s,e,w=find_bbox_coordinates(each_domain)
    north.append(n)
    south.append(s)
    east.append(e)
    west.append(w)

output_df=pd.DataFrame({
    "CORDEX_domain": domains,
    "north": north,
    "south": south,
    "east": east,
    "west": west
})
output_df.to_csv("cordex_domain_wgs4_coordinates.csv", index=False)



