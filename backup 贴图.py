
#究极大函数
from scipy.ndimage import shift
from astropy.io import fits
import os
import numpy as np
import observation_sim.psf.PSFInterp as PSFInterp
from observation_sim.instruments import Chip, Filter, FilterParam
import yaml
import galsim
import astropy.io.fits as fitsio
import matplotlib.pyplot as plt
from astropy.table import Table
import glob



exposure_ids = ["10109200100412","10109200100413","10109200100414"]
chip_ids = ["07","08","09","22","23","24"]
num = 200
exp_time = 150.
pupil_area = np.pi * (0.5 * 2)**2
CCD_size = 0.074
gain = 1.1


def load_psf(path):
    with fits.open(path) as hdul:
        return hdul[0].data  # Access the data in the primary HDU

# CSST params
# filter parameters: name: 
# 1) effective wavelength
# 2) effective width
# 3) blue end
# 4) red end
# 5) systematic efficiency:
#    tau = quantum efficiency * optical transmission * filter transmission
# 6) sky background: e/pix/s
# 7) saturation magnitude
# 8) dim end magnitude

filtP = {
        "NUV":   [2867.7,   705.4,  2470.0,   3270.0,  0.1404, 0.004, 15.7, 25.4],
        "u":     [3601.1,   852.1,  3120.0,   4090.0,  0.2176, 0.021, 16.1, 25.4],
        "g":     [4754.5,  1569.8,  3900.0,   5620.0,  0.4640, 0.164, 17.2, 26.3],
        "r":     [6199.8,  1481.2,  5370.0,   7030.0,  0.5040, 0.207, 17.0, 26.0],
        "i":     [7653.2,  1588.1,  6760.0,   8550.0,  0.4960, 0.212, 16.7, 25.9],
        "z":     [9600.6,  2490.5,  8240.0,  11000.0,  0.2000, 0.123, 15.7, 25.2],
        "y":     [10051.0, 1590.6,  9130.0,  11000.0,  0.0960, 0.037, 14.4, 24.4],

        "FGS":   [5000.0,  8000.0,  3000.0,  11000.0,  0.6500, 0.164, 0., 30.], # [TOD

        "GU":    [0.0,        0.0,  2550.0,   4200.0,     1.0, 0.037, 14.0, 26.0],
        "GV":    [0.0,        0.0,  4000.0,   6500.0,     1.0, 0.037, 14.0, 26.0],
        "GI":    [0.0,        0.0,  6200.0,  10000.0,     1.0, 0.037, 14.0, 26.0],
}
VC_A = 2.99792458e+18  # speed of light: A/s
VC_M = 2.99792458e+8   # speed of light: m/s
VC_KM= 2.99792458e+5   # speed of light: km/s
H_PLANK = 6.626196e-27 # Plank constant: erg s

def photonEnergy(lambd):
    """ The energy of photon at a given wavelength
    Parameter:
        lambd: the wavelength in unit of Angstrom
    Return:
        eph: energy of photon in unit of erg
    """
    nu = VC_A / lambd
    eph = H_PLANK * nu
    return eph


#get nphotons from given CSST_mag
def mag2photon(mag, filt=None, filtEfficiency=True):
    #magLim= filt.mag_limiting
    cband = filtP[filt][0]
    wband = filtP[filt][1]
    lowband = cband - 0.5*wband
    upband  = cband + 0.5*wband

    iflux = 10**(-0.4*(mag+48.6)) #mag2flux(mag) #erg/s/cm^2/Hz
    iphotonEnergy = photonEnergy(cband)
    nn = iflux/iphotonEnergy
    nn = nn * VC_KM*1e13*(1.0/lowband - 1.0/upband)
    nObsPhoton = nn   #photons/cm2/s


    if filtEfficiency == True:
        nObsPhoton = nObsPhoton * filtP[filt][4]
    nObsPhoton = nObsPhoton * 1e4 * pupil_area * exp_time
    return nObsPhoton

#图像处理部分（展开+切除）
def formatRevert(GSImage, nsecy=1, nsecx=16):
    img = GSImage
    ny, nx = img.shape
    dx = int(nx/nsecx)
    dy = int(ny/nsecy)

    newimg = galsim.Image(int(dx*8), int(dy*2), init_value=0)

    for ix in range(0, 4):
        tx = ix
        newimg.array[0:dy, 0+tx*dx:dx+tx*dx] = img[:, 0+ix*dx:dx+ix*dx]
    for ix in range(4, 8):
        tx = ix
        newimg.array[0:dy, 0+tx*dx:dx+tx*dx] = img[:, 0+ix*dx:dx+ix*dx][:, ::-1]
    for ix in range(8, 12):
        tx = 7-(ix-8)
        newimg.array[0+dy:dy+dy, 0+tx*dx:dx+tx*dx] = img[:, 0+ix*dx:dx+ix*dx][::-1, ::-1]
    for ix in range(12, 16):
        tx = 7-(ix-8)
        newimg.array[0+dy:dy+dy, 0+tx*dx:dx+tx*dx] = img[:, 0+ix*dx:dx+ix*dx][::-1, :]

    # 将 numpy 数组转换为 uint16 类型，并用 galsim.Image 的 set_array 方法重新设置
    newimg_uint16 = newimg.array.astype(np.uint16)
    newimg = galsim.Image(newimg_uint16)

    return newimg


def cut_image(GSImage):
    # Define the parameters for cutting the image
    px = 27
    ox = 71
    py = 0
    oy = 84
    dx = 1152
    dy = 4616
    
    # Define the regions to be removed
    remove_x = [
        (0, px),
        (dx + px, dx + 2 * px + ox),
        (2 * dx + 2 * px + ox, 2 * dx + 3 * px + 2 * ox),
        (3 * dx + 3 * px + 2 * ox, 3 * dx + 4 * px + 3 * ox),
        (4 * dx + 4 * px + 3 * ox, 4 * dx + 4 * px + 5 * ox),
        (5 * dx + 4 * px + 5 * ox, 5 * dx + 5 * px + 6 * ox),
        (6 * dx + 5 * px + 6 * ox, 6 * dx + 6 * px + 7 * ox),
        (7 * dx + 6 * px + 7 * ox, 7 * dx + 7 * px + 8 * ox),
        (8 * dx + 7* px + 8 * ox, 8 * dx + 8 * px + 8 * ox)
    ]

    
    remove_y = (dy, dy + 2 * oy)

    # Get the original image as a numpy array
    img_array = GSImage.array

    # Initialize a list to hold the remaining parts of the image
    remaining_parts_x = []

    # Start the cut process without updating coordinates each time
    current_start = 0
    
    # Iterate through the removal regions
    for start, end in remove_x:
        # Add the remaining part before the current removal region
        remaining_parts_x.append(img_array[:, current_start:start])
        # Update current_start to the end of the removed region
        current_start = end

    # Append the remaining part after the last removal region
    remaining_parts_x.append(img_array[:, current_start:])

    # Concatenate all remaining x parts
    remaining_x = np.concatenate(remaining_parts_x, axis=1)

    # Handle the y removal region
    # Initialize a list for the final image parts
    final_image_parts = []

    # Append the top part before the y removal
    final_image_parts.append(remaining_x[:remove_y[0], :])
    # Append the bottom part after the y removal
    final_image_parts.append(remaining_x[remove_y[1]:, :])

    # Concatenate the final image parts vertically
    final_image = np.concatenate(final_image_parts, axis=0)

    # Convert to uint16 and create a new galsim.Image object
    final_image_uint16 = final_image.astype(np.uint16)
    newimg = galsim.Image(final_image_uint16)

    return newimg


import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import glob
def PA_get(row,wcs):
    exposure_id = str(row['exposure_ID'])
    chip_id = str(row['ID_chip']).zfill(2)  
    # Get e1 and e2 from the current row
    e1_rd = row['e1']
    e2_rd = row['e2']

    # Calculate the PA in RA/Dec coordinates
    PA_RA_DEC = 0.5 * np.arctan2(e2_rd, e1_rd)

    # Convert PA to degrees
    PA_RA_DEC_deg = np.degrees(PA_RA_DEC)

    # Check if WCS contains the necessary attributes
    if hasattr(wcs.wcs, 'cdelt'):
        # Use cdelt to calculate rotation angle
        rotation_angle = np.arctan2(wcs.wcs.cdelt[1], wcs.wcs.cdelt[0])
        rotation_angle_deg = np.degrees(rotation_angle)

        # Convert PA from RA/Dec to pixel coordinates
        PA_pixel = PA_RA_DEC_deg - rotation_angle_deg

        # Ensure PA is between 0 and 180 degrees
        PA_pixel = PA_pixel % 180

        return PA_pixel
    else:
        print(f"Warning: Missing necessary WCS parameters in {fits_path}.")
        return np.nan  # or some default value





# Load the FITS file
def lensing_stamp (exposure_id,chip_id):
    file_path = f"./selected_target_{num}.fits"
    with fits.open(file_path) as hdul:
        # hdul.info()  # Display the structure of the FITS file
        arc_table = Table(hdul[1].data)  # Access the data in the first extension
        mask = np.where((arc_table['exposure_ID'] == str(exposure_id))&(arc_table['ID_chip'] == int(chip_id)))
        arc_data = arc_table[mask]
        arc =  hdul[2].data[mask]
    name_list = arc_data['obj_ID'].data
    a= len(name_list)
    print(f"exposure {exposure_id} chip {chip_id} has {a} sources")
    #PSF part
    # Define the variables
    
    # Setup PATH
    folder_path = f"/home/veciam/贴图"
    config_filename = folder_path + "/config_overall.yaml"
    chip_id = str(chip_id).zfill(2)
    exposure_id = str(exposure_id)
    # Construct the search pattern for the main FITS file
    search_pattern = os.path.join(folder_path,'天区1', exposure_id, f"*{exposure_id}_{chip_id}_L0_V01.fits")
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        raise FileNotFoundError(f"No FITS file found matching pattern: {search_pattern}")
    
    file_path = matching_files[0]
    base_dir = os.path.dirname(file_path)
     # Construct path for catalog file
    cat_pattern = os.path.join(base_dir, f"*{exposure_id}_{chip_id}_L0_V01.cat")
    matching_cat_files = glob.glob(cat_pattern)
    
    if not matching_cat_files:
        raise FileNotFoundError(f"No catalog file found matching pattern: {cat_pattern}")
    
    cat_filename = matching_cat_files[0]

    # Read cat file
    catFn = open(cat_filename, "r")
    line = catFn.readline()
    # print(cat_filename, '\n', line)
    imgPos = []
    chipID = -1
    for line in catFn:
        line = line.strip()
        columns = line.split()

        if chipID == -1:
            chipID = int(columns[1])
        else:
            assert chipID == int(columns[1])
        ximg = float(columns[3])
        yimg = float(columns[4])
        imgPos.append([ximg, yimg])
    imgPos = np.array(imgPos)



    # Read config file
    with open(config_filename, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            for key, value in config.items():
                # print(key + " : " + str(value))
                pass
        except yaml.YAMLError as exc:
            # print(exc)
            pass

    # Setup Chip
    chip = Chip(chipID=chipID, config=config)
    # print('chip.bound::', chip.bound.xmin, chip.bound.xmax,
          # chip.bound.ymin, chip.bound.ymax)

    cat_table = Table.read(cat_filename, format='ascii')
    obj_IDs = cat_table['obj_ID'].data
    indices = [np.where(obj_IDs == name)[0].tolist() for name in name_list]


    # Convert to a NumPy array
    nobj = np.concatenate(indices)



    for iobj in nobj:

        # print("\nget psf for iobj-", iobj, '\t', 'bandpass:', end=" ", flush=True)
        # Setup Position on focalplane
        # try get the PSF at some location (1234, 1234) on the chip
        x, y = imgPos[iobj, :]
        x = x+chip.bound.xmin
        y = y+chip.bound.ymin

        pos_img = galsim.PositionD(x, y)

        # Setup sub-bandpass
        # (There are 4 sub-bandpasses for each PSF sample)
        filter_param = FilterParam()
        filter_id, filter_type = chip.getChipFilter()
        filt = Filter(
            filter_id=filter_id,
            filter_type=filter_type,
            filter_param=filter_param,
            ccd_bandpass=chip.effCurve)
        bandpass_list = filt.bandpass_sub_list
        for i in range(len(bandpass_list)):
            # print(i, end=" ", flush=True)
            # say you want to access the PSF for the sub-bandpass at the blue end for that chip
            bandpass = bandpass_list[i]

            # Get corresponding PSF model
            psf_model = PSFInterp(chip=chip, npsf=100,
                                  PSF_data_file=config["psf_setting"]["psf_pho_dir"])
            psf = psf_model.get_PSF(
                chip=chip, pos_img=pos_img, bandpass=bandpass, galsimGSObject=False)

            # if True:
            #     fn = "psf_{:}.{:}.{:}.fits".format(chipID, iobj, i)
            #     if fn != None:
            #         if os.path.exists(fn):
            #             os.remove(fn)
            #     hdu = fitsio.PrimaryHDU()
            #     hdu.data = psf
            #     hdu.header.set('pixScale', 5)
            #     hdu.writeto(fn)
            

            # Define a base output directory
            output_dir = f"{folder_path}/天区1/{exposure_id}"

            if True:
                # Create subdirectories based on chipID
                chip_dir = os.path.join(output_dir, f"chip_{chip_id}")
                os.makedirs(chip_dir, exist_ok=True)

                # Construct the filename
                fn = os.path.join(chip_dir, f"psf_{chip_id}_{iobj}_{i}.fits")

                # Remove the file if it already exists
                if os.path.exists(fn):
                    os.remove(fn)

                # Create and write the FITS file
                hdu = fitsio.PrimaryHDU()
                hdu.data = psf
                hdu.header.set('pixScale', 5)
                hdu.writeto(fn,overwrite=True)





    with fits.open(file_path) as hdul:
        image_data = hdul[1].data  # 提取数据
        header = hdul[0].header
        wcs = WCS(header)
    image_2_8 = formatRevert(image_data, nsecy=1, nsecx=16)
    mock_sky = cut_image(image_2_8)
    sky_array = mock_sky.array.astype(np.float32)  # 先转换为float32进行计算


    for index, obj in enumerate(nobj):


        bandPass = 'g' if chip_id in [8, 23] else 'i' if chip_id in [7, 24] else 'r' if chip_id in [9, 22] else 'z'

        mag=arc_data['mag'][index]
        nphoton= mag2photon(mag, filt=bandPass, filtEfficiency=True)
        # print("nphoton::", nphoton)

        # Create InterpolatedImage from arc data
       
        PA = PA_get(arc_data[index],wcs)
        
        arc_img = galsim.InterpolatedImage(galsim.ImageF(arc[index], scale=0.074), flux=nphoton)  
        arc_img = arc_img.rotate(theta=(PA-90)*galsim.degrees)
        psf_data = []
        for i in range(4):
            psf_path = f"{folder_path}/天区1/{exposure_id}/chip_{chip_id}/psf_{chip_id}_{obj}_{i}.fits"
            psf_data.append(load_psf(psf_path))
        # Combine the 4 PSF data
        combined_psf = np.sum(psf_data, axis=0) / 4
        # Create a larger array to hold the PSF data
        psfTemp = np.zeros([257, 257])
        psfTemp[0:256, 0:256] = combined_psf
        # Create InterpolatedImage from the combined PSF
        psf_img = galsim.InterpolatedImage(galsim.ImageF(psfTemp, scale=0.037))

        # Convolve the arc image with the PSF
        psf_arc = galsim.Convolve([arc_img, psf_img])

        # Draw the convolved image
        #psf_arc_image = psf_arc.drawImage(nx=64, ny=64, scale=0.074)
        #加入泊松
        stamp = galsim.ImageF(ncol=100, nrow=100, scale=CCD_size) #######
        stamp.setCenter(50,50)  #######
        sensor = galsim.Sensor()

        # Draw the convolved image with photon shooting and apply sensor effects
        ###final_image = psf_arc.drawImage(nx=64, ny=64, scale=0.074, method='phot', n_photons=nphoton, sensor=sensor)
        final_image = psf_arc.drawImage(method='phot', save_photons=True).photons
        final_image.x= final_image.x+50 ##############
        final_image.y= final_image.y+50 ##############
        sensor.accumulate(final_image, stamp)
    #     random_seed = galsim.BaseDeviate(1512413).raw()
    #     rng = galsim.UniformDeviate(random_seed+index+1)
    #     final_image = psf_arc.drawImage(nx=64, ny=64, scale=0.074)

        x_image = arc_data[index]['xImage']
        y_image = arc_data[index]['yImage']
        x_nominal = int(np.floor(x_image + 0.5))
        y_nominal = int(np.floor(y_image + 0.5))
        dx = x_image - x_nominal
        dy = y_image - y_nominal
            # 1. 转换为数组并确保数据类型

        stamp_array = stamp.array.astype(np.float32)/gain

        # 2. 对stamp进行亚像素偏移
        shifted_stamp = shift(stamp_array, (dy, dx), order=3)  # order=3 使用三次样条插值

        # 3. 计算要粘贴的位置（中心位置）
        x_start = x_nominal - shifted_stamp.shape[1]//2
        y_start = y_nominal - shifted_stamp.shape[0]//2

        # 4. 计算结束位置
        x_end = x_start + shifted_stamp.shape[1]
        y_end = y_start + shifted_stamp.shape[0]

        # 5. 确保不超出边界
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(sky_array.shape[1], x_end)
        y_end = min(sky_array.shape[0], y_end)

        # 6. 计算对应的stamp区域
        stamp_y_start = max(0, -y_start)
        stamp_x_start = max(0, -x_start)
        stamp_y_end = shifted_stamp.shape[0] - max(0, y_end - sky_array.shape[0])
        stamp_x_end = shifted_stamp.shape[1] - max(0, x_end - sky_array.shape[1])

        # 7. 直接相加（现在两个数组都是float32类型）
        sky_array[y_start:y_end, x_start:x_end] += shifted_stamp[stamp_y_start:stamp_y_end, stamp_x_start:stamp_x_end]


    sky_array = sky_array.astype(np.uint16)
    combined_image = galsim.Image(sky_array, bounds=mock_sky.bounds)



    # 创建一个 HDU 列表
    hdu_list = []

    # 将合并的图像添加到 HDU 列表
    image_hdu = fits.PrimaryHDU(data=combined_image.array)
    image_hdu.header['DESCRIPTION'] = 'Combined image'
    hdu_list.append(image_hdu)


    output_folder = f"combined_image/{exposure_id}"
    # 创建新的文件夹，如果不存在的话
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 创建一个 FITS 文件并写入 HDU 列表
    fits_file_path = f"{folder_path}/{output_folder}/combined_image_and_arc_data_{exposure_id}_{chip_id}.fits"
    fits.HDUList(hdu_list).writeto(fits_file_path, overwrite=True)  # 如果文件已存在，覆盖它

    print(f"Combined image and arc data saved to: {fits_file_path}")
    
from mpi4py import MPI
import socket

# 定义曝光和芯片ID
# exposure_ids = ["10109200100412", "10109200100413", "10109200100414",  # 示例曝光ID
#                 # 这里可以继续添加更多曝光ID
#                ]

# 初始化MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

# 计算曝光ID的总数
num_exposures = len(exposure_ids)

# 在每个进程中处理曝光ID
for i in range(comm_rank, num_exposures, comm_size):
    exposure_id = exposure_ids[i]
    print(f"Process {comm_rank} processing exposure {exposure_id}")

    ### mkdir("outputs/"+exposure_id)
    
    # 对每个曝光ID串行处理芯片ID
    for chip_id in chip_ids:
        print(f"Process {comm_rank} processing chip {chip_id} for exposure {exposure_id}")
        lensing_stamp(exposure_id, chip_id)  # outPath="outputs/"+exposure_id

    print(f"Process {comm_rank} finished processing exposure {exposure_id}")