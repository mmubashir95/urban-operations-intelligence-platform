"""Configuration for NYC 311 Open Data notebook analysis."""

from urllib.parse import urlencode

API_ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
API_TIMEOUT_SECONDS = 180
API_MAX_ATTEMPTS = 3
SAMPLE_ROWS_PER_YEAR = 1_000
GROUP_RESULT_LIMIT = 50_000

SELECTED_COLUMNS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "descriptor_2",
    "location_type",
    "incident_zip",
    "incident_address",
    "street_name",
    "cross_street_1",
    "cross_street_2",
    "intersection_street_1",
    "intersection_street_2",
    "address_type",
    "city",
    "landmark",
    "facility_type",
    "status",
    "due_date",
    "resolution_description",
    "resolution_action_updated_date",
    "community_board",
    "council_district",
    "police_precinct",
    "bbl",
    "borough",
    "x_coordinate_state_plane",
    "y_coordinate_state_plane",
    "open_data_channel_type",
    "park_facility_name",
    "park_borough",
    "vehicle_type",
    "taxi_company_borough",
    "taxi_pick_up_location",
    "bridge_highway_name",
    "bridge_highway_direction",
    "road_ramp",
    "bridge_highway_segment",
    "latitude",
    "longitude",
    "location",
]

SOURCE_QUERY = "SELECT " + ", ".join(SELECTED_COLUMNS)
SOURCE_API_URL = f"{API_ENDPOINT}?{urlencode({'$query': SOURCE_QUERY})}"
