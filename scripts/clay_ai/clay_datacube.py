#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch
from clay.module import ClayMAEModule

"""
Clay Model expects a dictionary with keys:

pixels: batch x band x height x width - normalized chips of a sensor
time: batch x 4 - horizontally stacked week_norm & hour_norm
latlon: batch x 4 - horizontally stacked lat_norm & lon_norm
waves: list[:band] - wavelengths of each band of the sensor from the metadata.yaml file
gsd: scalar - gsd of the sensor from metadata.yaml file
"""

# Specify the columns to import the raw pixels data using their names in header

filepath = "../Model_A+_Soil+Sentinel_9x9.csv"


data_header = pd.read_csv(filepath, sep=",", nrows=0)
print(data_header.shape)
idx_start = data_header.columns.get_loc("B01_1_normalized")
idx_end = data_header.columns.get_loc("B12_81_normalized")
idx_lat = data_header.columns.get_loc("TH_LAT")
idx_long = data_header.columns.get_loc("TH_LONG")
print(idx_lat, idx_long, idx_start, idx_end)

# read the raw data from csv file
data_lat = pd.read_csv(filepath, sep=",", usecols=[idx_lat], header=0)
data_long = pd.read_csv(filepath, sep=",", usecols=[idx_long], header=0)
data_bands = pd.read_csv(
    filepath, sep=",", usecols=range(idx_start, idx_end + 1), header=0
)
print(data_bands.shape)


# latlon: batch x 4 - horizontally stacked lat_norm & lon_norm
latlong = pd.DataFrame()
latlong["lat_sin"] = np.sin(data_lat * np.pi / 180)
latlong["lat_cos"] = np.cos(data_lat * np.pi / 180)
latlong["long_sin"] = np.sin(data_long * np.pi / 180)
latlong["long_cos"] = np.cos(data_long * np.pi / 180)
print(latlong.head, latlong.shape)

latlong_clay = latlong.to_numpy()


# time: batch x 4 - horizontally stacked week_norm & hour_norm
# time is set to zero, since we don't consider the timestamps in the format of week and hour
time = pd.DataFrame(np.zeros((data_bands.shape[0], 4)))
print(time.head, time.shape)

time_clay = time.to_numpy()


"""
pixels: batch x band x height x width - normalized chips of a sensor
we use 12 bands with pro chip 9x9 pixels and the data of pixels is already normalized imported


list_bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]

"""
m = 81  # shape = (9x9) -> 81 pixel each band
data_bands_np = data_bands.to_numpy()
data_bands_clay = np.zeros((data_bands_np.shape[0], 12, 9, 9))
for i in range(0, data_bands_np.shape[0]):
    for j in range(0, 12):
        data_bands_clay[i, j, :, :] = data_bands_np[
            i, (0 + m * j) : (m + m * j)
        ].reshape(9, 9)
print(data_bands_clay[0:1], data_bands_clay.shape)


# load the Clay Model
"""
load the model from check point
wget -q https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ckpt = "clay-v1.5.ckpt"
torch.set_default_device(device)

model = ClayMAEModule.load_from_checkpoint(
    ckpt,
    model_size="large",
    # the model size can only be "large", because the weights and bias in clay-v1.5ckpt are trained with large model with kernel size 8x8
    # metadata_path="../../configs/metadata.yaml",
    dolls=[16, 32, 64, 128, 256],
    doll_weights=[1, 1, 1, 1, 1],
    mask_ratio=0.0,
    shuffle=False,
)
model.eval()

model = model.to(device)


# Prepare additional information of wavelength and gsd

"""
    wavelength:
      water quality: 0.443
      blue: 0.493
      green: 0.56
      red: 0.665
      rededge1: 0.704
      rededge2: 0.74
      rededge3: 0.783
      nir: 0.842
      nir08: 0.865
      water vapor: 0.945
      swir16: 1.61
      swir22: 2.19
"""
gsd = 60
waves = [0.443, 0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 0.945, 1.61, 2.19]
platform = "sentinel-2-l2a"


# pack all the data into a data cube in form of a dictionary as input for clay
datacube = {
    "platform": platform,
    "time": torch.tensor(
        time_clay,
        dtype=torch.float32,
        device=device,
    ),
    "latlon": torch.tensor(latlong_clay, dtype=torch.float32, device=device),
    "pixels": torch.tensor(data_bands_clay, dtype=torch.float32, device=device),
    "gsd": torch.tensor(gsd, device=device),
    "waves": torch.tensor(waves, device=device),
}

# run the clay model for the embeddings using the data cube above

with torch.no_grad():
    unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)

# The first embedding is the class token, which is the overall single embedding. We extract that for our research.
embeddings = unmsk_patch[:, 0, :].cpu().numpy()
print(embeddings.shape)

# save the embeddings as output in a csv file
np.savetxt("embeddings.csv", embeddings, delimiter=",", fmt="%f")
