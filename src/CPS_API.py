import os
import glob
from datetime import datetime
from pathlib import Path
from ipumspy import IpumsApiClient, MicrodataExtract, readers, ddi
from pandas import read_csv

# Should replace with your own key
IPUMS_API_KEY = "59cba10d8a5da536fc06b59d3b7781dc64fe4a1ea7c19179c8a6e41b"

start_date = datetime(2022, 1, 1)
end_date = datetime(2025, 3, 31)

ipums = IpumsApiClient(IPUMS_API_KEY)

samples = []
codes = read_csv(os.path.join(os.path.dirname(__file__), "..", "resources/sample_codes.csv"), index_col = 0)

# Generate all the monthly samples from Jan 2022 to Mar 2025
current_date = start_date
while current_date <= end_date:
    sample_name = f"cps{current_date.year}_{current_date.month:02d}"
    samples.append(sample_name)

    # Move to next month
    if current_date.month == 12:
        current_date = datetime(current_date.year + 1, 1, 1)
    else:
        current_date = datetime(current_date.year, current_date.month + 1, 1) 

samples = codes.loc[samples, "Sample ID"]

extract = MicrodataExtract(
    collection = "cps",
    description = "Test CPS Extract",
    samples = samples,
    variables = [
        "SERIAL",    # Household serial number
        "MONTH",     # Month
        "YEAR",      # Year
        "CPSID",     # CPSID, Unique person identifier
        "PERNUM",    # Person number in sample unit
        "WTFINL",    # Final basic weight
        "AGE",       # Age
        "SEX",       # Sex
        "RACE",      # Race
        "HISPAN",    # Hispanic origin
        "EDUC",      # Educational attainment
        "EMPSTAT",   # Employment status
        "LABFORCE",  # Labor force status
        "UHRSWORKT", # Hours usually worked per week
        "EARNWEEK",  # Weekly earnings
        "HOURWAGE",  # Hourly wage
        "OCC",       # Occupation
        "IND",       # Industry
        "STATEFIP",  # State (FIPS code)
        "METRO",     # Metropolitan area status
    ]
)

# Submit the extract request
ipums.submit_extract(extract)
print(f"Extract submitted with id {extract.extract_id}")

# Wait for the extract to finish
ipums.wait_for_extract(extract)

print(extract)

# Download the extract
DOWNLOAD_DIR = Path("./resources/cps_extracts")
ipums.download_extract(extract, download_dir=DOWNLOAD_DIR)

# Get the DDI
ddi_file = list(DOWNLOAD_DIR.glob("*.xml"))[0]
ddi = readers.read_ipums_ddi(ddi_file)

# Get the data
cps_data = readers.read_microdata(ddi, DOWNLOAD_DIR / ddi.file_description.filename)

# Now you can do your analysis