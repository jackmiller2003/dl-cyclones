
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap(ds: 'xarray.DataArray', time=0, level=0):
    x = ds.isel(time=time, level=level).to_numpy()
    sns.heatmap(x)
    plt.show()
