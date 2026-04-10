import urllib.request
import zipfile
import os

url = 'https://github.com/nikivanstein/EoH/archive/f0d21b2f2f62c452bac545fd7f68d3af50845321.zip'
print("Downloading...")
urllib.request.urlretrieve(url, 'EoH.zip')
print("Extracting...")
with zipfile.ZipFile('EoH.zip', 'r') as zip_ref:
    zip_ref.extractall('EoH_extract')
print("Done.")
