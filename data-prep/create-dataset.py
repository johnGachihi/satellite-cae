import fiona
from fiona import transform
import geopandas as gpd
import pyogrio
import earthaccess
import rioxarray as rxr
import rasterio
import xarray as xr
from shapely.geometry import shape, mapping, Point
import numpy as np
import sys

s2_bands = ["B02", "B03", "B04", "B8A", "B11", "B12"]
l_bands  = ["B02", "B03", "B04", "B05", "B06", "B07"]

def is_s2_link(band_link):
  return band_link.split("/")[4] == "HLSS30.020"

def filter_bands(granules):
    filtered = []
    for granule in granules:
        filtered_links = []
        bands_to_select = s2_bands if is_s2_link(granule[0]) else l_bands
        for link in granule:
            if any(band in link for band in bands_to_select):
                filtered_links.append(link)
        filtered.append(filtered_links)
    return filtered

def get_band(link):
    return link.rsplit(".", 2)[-2]

def get_datetime(link):
    return link.rsplit(".", 5)[-5]

def clip(raster, polygon):
    raster_crs = raster.spatial_ref.crs_wkt
    polygon = fiona.transform.transform_geom(SAMPLES_CRS, raster_crs, mapping(polygon))

    center = shape(polygon).centroid
    transform = raster.rio.transform()
    x, y = ~transform * (center.x, center.y)
    x, y = int(np.round(x)), int(np.round(y))

    half_window = int(IMAGE_SIZE / 2)
    start_row = y - half_window
    end_row = y + half_window
    start_col = x - half_window
    end_col = x + half_window

    raster_end_row = raster.shape[0] - 1
    raster_end_col = raster.shape[1] - 1

    if start_col < 0:
        diff = 0 - start_col
        start_col += diff
        end_col += diff
    elif end_col > raster_end_col:
        diff = end_col - raster_end_col
        end_col -= diff
        start_col -= diff

    if start_row < 0:
        diff = 0 - start_row
        start_row += diff
        end_row += diff
    elif end_row > raster_end_row:
        diff = end_row - raster_end_row
        end_row -= diff
        start_row -= diff

    new_center = ((end_col - start_col) / 2, (end_row - start_row) / 2)
    new_center = transform * new_center
    new_center = Point(new_center)
    new_center = fiona.transform.transform_geom(raster_crs, SAMPLES_CRS, mapping(new_center))
    new_center = shape(new_center)

    return raster[start_row:end_row, start_col:end_col], new_center


def scaling(band):
    scale_factor = band.attrs['scale_factor']
    band_out = band.copy()
    band_out.data = band.data*scale_factor
    band_out.attrs['scale_factor'] = 1
    return(band_out)


class InvalidRasterException(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_fmask_url(band_urls):
    url = band_urls[0]
    url = url.rsplit(".", 2)
    url[-2] = "Fmask"
    return ".".join(url)


FMASK_CLOUD = 1
FMASK_ADJ_TO_CLOUD_OR_SHADOW = 2
FMASK_CLOUD_SHADOW = 3

def assert_quality(fmask):
    total_pixels = np.prod(fmask.shape)

    # nans
    nan_count = fmask.isnull().sum().compute()
    nan_perc = nan_count / total_pixels
    if nan_perc > 0.1:
        raise InvalidRasterException(f"Too much missing data ({(nan_perc * 100):.2f}%)")

    # clouds / cloud-shadow / adj to cloud/shadow
    cloud_count = ((fmask == FMASK_CLOUD) |
                   (fmask == FMASK_ADJ_TO_CLOUD_OR_SHADOW) |
                   (fmask == FMASK_CLOUD_SHADOW)
                  ).sum().compute()
    cloud_perc = cloud_count /total_pixels
    if cloud_perc > 0.3:
        raise InvalidRasterException(f"Too much cloud/shadow ({(cloud_perc * 100):.2f}%)")


def load_multiband_raster(img_band_urls, region):
    with rasterio.Env(GDAL_HTTP_COOKIEFILE='~/cookies.txt', GDAL_HTTP_COOKIEJAR='~/cookies.txt'):
        chunk_size = {"band": 1, "x": 512, "y": 512}

        fmask_url = get_fmask_url(img_band_urls)
        fmask = rxr.open_rasterio(fmask_url, chunks=chunk_size, masked=True).squeeze("band", drop=True)
        fmask, _ = clip(fmask, region)
        assert_quality(fmask)

        bands = []
        band_names = []
        for band_url in sorted(img_band_urls, key=get_band):
            raster = rxr.open_rasterio(band_url, chunks=chunk_size, masked=True).squeeze("band", drop=True)
            raster.attrs['scale_factor'] = 0.0001
            raster.attrs['long_name'] = get_band(band_url)
            raster, center = clip(raster, region)
            raster = scaling(raster)
            bands.append(raster)
            band_names.append(get_band(band_url))

        stacked_bands = xr.concat(bands, dim="band")
        stacked_bands = stacked_bands.assign_coords(band=band_names)
        stacked_bands = stacked_bands.sel(
            band=s2_bands if is_s2_link(img_band_urls[0]) else l_bands)

        return stacked_bands, center

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python create-dataset.py <samples-file.shp> <out dir>")
  else:
    earthaccess.login(persist=True)

    samples_file = sys.argv[1]

    SAMPLES_CRS = "EPSG:4326"
    IMAGE_SIZE = 512

    img_sqrs = gpd.read_file(samples_file, engine="pyogrio")
    img_sqrs.set_crs(SAMPLES_CRS)

    data_dir = sys.argv[2]

    for idx, sqr in enumerate(img_sqrs["geometry"]):
      search_res = earthaccess.search_data(
        short_name=['HLSL30','HLSS30'],
        bounding_box=sqr.bounds,
        temporal=("2023-01-01","2023-02-27"),
        count=-1,
        cloud_cover=(0, 30) # ***
      )
      imgs_urls = [granule.data_links() for granule in search_res]
      imgs_urls = filter_bands(imgs_urls)

      for _idx, img_urls in enumerate(imgs_urls):
        img_date = get_datetime(img_urls[0])

        try:
          img, center = load_multiband_raster(img_urls, sqr)
        except InvalidRasterException as e:
          print("Invalid raster encountered:", e)
          continue

        img = img.astype("float32")
        img_name = f"{img_date}_{center.x:.3f}_{center.y:.3f}.tif"
        img.rio.to_raster(f"{data_dir}/{img_name}")

        print(f"Image {img_name} done")

