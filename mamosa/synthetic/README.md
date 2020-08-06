# Synthetic data

Class for making synthetic geo-volumes with horizons and faults, and for
generating synthetic seismic from the corresponding geo-volumes.

## Usage
Example:

```python
from mamosa.synthetic import SyntheticData
from mamosa.utils.plot_utils import (
    imshow_seismic,
    imshow_shifted_seismic,
    plot_horizon_3d,
)
import numpy as np

np.random.seed(123)

synth = SyntheticData((128, 128, 128))
horizons = synth.generate_horizons(
    n_horizons=4, min_distance=3, fault_xlines=[30, 80], fault_size=10
)
systematic_sigma = 0.02
white_sigma = 0.002
blur_sigma = 0.5
seismic = synth.generate_synthetic_seismic(
    systematic_sigma=systematic_sigma, white_sigma=white_sigma, blur_sigma=blur_sigma
)
i = 50
imshow_seismic(seismic[i], horizons[2][i], size=5)
imshow_shifted_seismic(seismic[i], horizons[2][i])
plot_horizon_3d(horizons[2], seismic, i=0, x=0, view_init=(20, 45))

# Get horizon 2 in binary 3d format
horizon_3d = synth.horizon_volume(2)
print(horizon_3d.shape)  # (128, 128, 128)
print(horizon_3d.sum())  # 128**2 = 16384
```
