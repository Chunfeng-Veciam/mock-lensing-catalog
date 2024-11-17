from typing import List, Dict, Set, Union
from collections import Counter
import os
from astropy.table import Table, vstack
import numpy as np
base_path = "/home/veciam/贴图/天区1"
exposure_ids = [10109200100412,10109200273683]
num = 70
epsilon_vdisp = 10  # Example epsilon for velocity dispersion
epsilon_mag = 0.5   # Example epsilon for magnitude
epsilon_q = 0.1

best_band = lens_samples['best_band'].data #band with highest SNR of lensed arc
magnification = lens_samples['mu'].data
#lens magnitudes in [griz] band
lens_mag_g = lens_samples['mag_g_l'].data
lens_mag_r = lens_samples['mag_r_l'].data
lens_mag_i = lens_samples['mag_i_l'].data
lens_mag_z = lens_samples['mag_z_l'].data
#source magnitudes in [griz] band
src_mag_g = lens_samples['mag_g_s0'].data
src_mag_r = lens_samples['mag_r_s0'].data
src_mag_i = lens_samples['mag_i_s0'].data
src_mag_z = lens_samples['mag_z_s0'].data
# %%
#----------------use lensing catalog to generate mock images----------
from SimCsstLens.SimLensImage.MockSurvey import MockSurvey
from SimCsstLens.SimLensImage import Util as SSU

class NewMockSurvey(MockSurvey):
    def load_ideal_lens_from_table(self, this_table):
        self.src_z = this_table['z_s0'].data.reshape(1, -1) #redshift
        self.src_xs = this_table['x_s0'].data.reshape(1, -1) #src x-position in arcsec
        self.src_ys = this_table['y_s0'].data.reshape(1, -1) #src y-position in arcsec
        self.src_Re = this_table['re_s0'].data.reshape(1, -1) #src effective radius in arcsec
        self.src_q = this_table['q_s0'].data.reshape(1, -1) #src axis ratio
        self.src_pa = this_table['pa_s0'].data.reshape(1, -1) #src position angle in degree
        self.src_thetaE = this_table['thetaE_s0'].data.reshape(1, -1) #Einstein radius in arcsec
        
        self.dfl_Re = this_table['re_l'].data #lens light effective radius in arcsec
        self.dfl_z = this_table['z_l'].data #lens redshift
        self.dfl_q = this_table['q_l'].data #lens axis ratio, for both light and mass
        self.dfl_vdisp = this_table['vdisp_l'].data #lens velocity dispersion in km/s
        self.mu = this_table['mu'].data #lens magnification
        for band in self.bands[0:-1]:
            self.__dict__[f'src_app_mag_{band}'] = this_table[f'mag_{band}_s0'].data.reshape(1, -1) #src apparent magnitude in the given band
            self.__dict__[f'src_app_mag_{band}'] = self.__dict__[f'src_app_mag_{band}'] - 2.5 * np.log10(self.mu) #correction term
            self.__dict__[f'dfl_app_mag_{band}'] =  this_table[f'mag_{band}_l'].data #lens light apparent magnitude in the given band
        self.n_ideal_lenses = len(self.dfl_z)
        
from SimCsstLens.SimLensImage.MockSurvey import MockSurvey
survey = NewMockSurvey(config_path="./", config_file='csst_setting.yaml')
survey.load_ideal_lens_from_table(lens_samples)

def read_and_combine_cat_files(base_path, exposure_ids):
    # Initialize a list to store the individual tables
    cat_tables = []

    # Loop through each exposure ID
    for exposure_id in exposure_ids:
        # Construct the folder path using the base path and exposure ID
        folder_path = os.path.join(base_path, str(exposure_id))

        # Check if the directory exists
        if os.path.isdir(folder_path):
            # Get a list of all .cat files in the current folder
            cat_files = [f for f in os.listdir(folder_path) if f.endswith('.cat')]

            # Loop through all .cat files and read them
            for cat_file in cat_files:
                # Extract the exposure ID from the filename
                parts = cat_file.split('_')
                file_exposure_id = parts[-4]  # The second to last part is the exposure ID

                # Read the .cat file
                file_path = os.path.join(folder_path, cat_file)
                cat_table = Table.read(file_path, format='ascii')

                # Add the exposure_ID column to the table
                cat_table['exposure_ID'] = file_exposure_id

                cat_tables.append(cat_table)

    # Combine all tables into a single big table if any tables were read
    if cat_tables:
        combined_cat = vstack(cat_tables)
        return combined_cat
    else:
        # Return an empty table if no valid files were found
        return Table()

def filter_combined_cat(combined_cat):
    """
    Filters the combined catalog based on specified criteria.
    
    Parameters:
    combined_cat (Table): The combined catalog to filter.
    
    Returns:
    Table: A filtered catalog with only valid targets.
    """
    # Step 1: Create masks based on the criteria
    mask_not_star = combined_cat['galType'] != -1  # Not a star
    mask_not_dim = combined_cat['mag'] <= 24  # Not dim
    mask_not_boundary = (combined_cat['xImage'] > 128) & (combined_cat['xImage'] < 9188) & \
                        (combined_cat['yImage'] > 128) & (combined_cat['yImage'] < 9104)

    # Step 2: Combine all masks
    combined_mask = mask_not_star & mask_not_dim & mask_not_boundary

    # Step 3: Identify obj_IDs that do not satisfy the criteria
    invalid_obj_ids = combined_cat['obj_ID'][~combined_mask]

    # Step 4: Create a final mask that excludes all rows with invalid obj_IDs
    final_mask = np.isin(combined_cat['obj_ID'], invalid_obj_ids, invert=True)

    # Step 5: Apply the final mask to get the filtered table
    filtered_combined_cat = combined_cat[final_mask]

    return filtered_combined_cat



def cross_match_lensing_improved(final_table, lensing_table):
    """
    Improved version that handles duplicate obj_IDs by adding a unique identifier
    """
    cross_matched_dict = {}
    unmatched_obj_ids = set()
    
    # Add a unique identifier to each row
    for i, row in enumerate(final_table):
        # Extract parameters as before
        vdisp = row['veldisp']
        e1 = row['e1']
        e2 = row['e2']
        e = np.sqrt(e1**2 + e2**2)
        q = (1 - e) / (1 + e)
        
        # Create unique key combining obj_ID and row index
        unique_key = f"{row['obj_ID']}_{i}"
        
        # Rest of matching logic...
        id_chip = row['ID_chip']
        if id_chip in [8, 23]:
            mag = row['mag']
            mag_column = 'mag_g_l'
        elif id_chip in [7, 24]:
            mag = row['mag']
            mag_column = 'mag_i_l'
        elif id_chip in [9, 22]:
            mag = row['mag']
            mag_column = 'mag_r_l'
        else:
            continue

        # Matching logic
        mask_vdisp = np.abs(lensing_table['vdisp_l'] - vdisp) < epsilon_vdisp
        mask_mag = np.abs(lensing_table[mag_column] - mag) < epsilon_mag
        mask_q = np.abs(lensing_table['q_l'] - q) < epsilon_q
        
        match_mask = mask_vdisp & mask_mag & mask_q
        matched_indices = np.where(match_mask)[0]

        if matched_indices.size > 0:
            cross_matched_dict[unique_key] = {
                'obj_ID': row['obj_ID'],
                'source_index': i,  # Store original index
                'ID_chip': row['ID_chip'],
                'ra_orig': row['ra_orig'],
                'dec_orig': row['dec_orig'],
                'z': row['z'],
                'mag': row['mag'],
                'vdisp': vdisp,
                'q': q,
                'e1': e1,
                'e2': e2,
                'exposure_ID': row['exposure_ID'],
                'xImage': row['xImage'],
                'yImage': row['yImage'],
                'lens_sample_rows': matched_indices.tolist(),
            }
        else:
            unmatched_obj_ids.add(unique_key)

    # Create the output table
    cross_matched_rows = []
    for unique_key, data in cross_matched_dict.items():
        cross_matched_rows.append({
            'obj_ID': data['obj_ID'],
            'source_index': data['source_index'],
            'ID_chip': data['ID_chip'],
            'ra_orig': data['ra_orig'],
            'dec_orig': data['dec_orig'],
            'z': data['z'],
            'mag': data['mag'],
            'vdisp': data['vdisp'],
            'q': data['q'],
            'e1': data['e1'],
            'e2': data['e2'],
            'exposure_ID': data['exposure_ID'],
            'xImage': data['xImage'],
            'yImage': data['yImage'],
            'lens_sample_rows': data['lens_sample_rows']
        })

    # Convert to Table
    matched_table = Table(rows=cross_matched_rows) if cross_matched_rows else None

    # Filter final table
    if unmatched_obj_ids:
        # Convert unique keys back to obj_IDs for filtering
        unmatched_simple_ids = {key.split('_')[0] for key in unmatched_obj_ids}
        final_mask = ~np.isin(final_table['obj_ID'], list(unmatched_simple_ids))
        filtered_final_table = final_table[final_mask]
        if matched_table is not None:
            mask = ~np.isin(matched_table['obj_ID'], list(unmatched_simple_ids))
            matched_table = matched_table[mask]
    else:
        filtered_final_table = final_table

    return matched_table, filtered_final_table




def filter_and_apply_overlap(matched_table: Table) -> Table:
    """
    Filter and process rows in an Astropy Table based on obj_ID and lens_sample_rows overlap.
    
    This function groups rows by obj_ID, finds overlapping lens_sample_rows within each group,
    and applies a consistent sample across rows with the same obj_ID where overlap exists.
    
    Parameters
    ----------
    matched_table : astropy.table.Table
        Input table containing at least 'obj_ID' and 'lens_sample_rows' columns.
        'lens_sample_rows' should be a list-like object for each row.
    
    Returns
    -------
    astropy.table.Table
        Filtered table where:
        - Groups with overlap have a single common sample chosen
        - Single rows are preserved
        - Groups without overlap are excluded
    
    Raises
    ------
    KeyError
        If required columns ('obj_ID' or 'lens_sample_rows') are missing
    ValueError
        If the input table is empty or if lens_sample_rows contains invalid data
    """
    # Input validation
    if len(matched_table) == 0:
        raise ValueError("Input table is empty")
    
    required_columns = {'obj_ID', 'lens_sample_rows'}
    if not all(col in matched_table.columns for col in required_columns):
        raise KeyError(f"Input table must contain columns: {required_columns}")
    
    # Create a dictionary to hold rows by obj_ID
    grouped_rows: Dict[str, List[Dict]] = {}
    
    # Group rows by obj_ID
    for row in matched_table:
        try:
            obj_ID = str(row['obj_ID'])  # Convert to string for consistent handling
            if not isinstance(row['lens_sample_rows'], (list, np.ndarray, set)):
                raise ValueError(f"Invalid lens_sample_rows format for obj_ID {obj_ID}")
            
            if obj_ID not in grouped_rows:
                grouped_rows[obj_ID] = []
            grouped_rows[obj_ID].append(dict(row))
        except Exception as e:
            raise ValueError(f"Error processing row with obj_ID {obj_ID}: {str(e)}")
    
    filtered_rows = []
    
    # Process each group
    for obj_ID, rows in grouped_rows.items():
        try:
            # Create sets of lens_sample_rows
            lens_samples_sets: List[Set] = [
                set(row['lens_sample_rows']) if isinstance(row['lens_sample_rows'], (list, np.ndarray)) 
                else row['lens_sample_rows'] 
                for row in rows
            ]
            
            # Find intersection if multiple rows exist
            if len(lens_samples_sets) > 1:
                common_samples = set.intersection(*lens_samples_sets)
            else:
                common_samples = lens_samples_sets[0]
            
            if common_samples:
                # Choose one sample from the overlap
                chosen_sample = np.random.choice(list(common_samples))
                
                # Apply chosen sample to all rows in group
                for row in rows:
                    row['lens_sample_rows'] = [chosen_sample]
                filtered_rows.extend(rows)
            elif len(rows) == 1:
                # Keep single rows regardless of overlap
                filtered_rows.extend(rows)
            # Multiple rows without overlap are excluded
            
        except Exception as e:
            raise ValueError(f"Error processing group with obj_ID {obj_ID}: {str(e)}")
    
    # Convert back to Table
    try:
        return Table(rows=filtered_rows)
    except Exception as e:
        raise ValueError(f"Error creating output table: {str(e)}")
        
        

def check_duplicates(table):
    """
    Simply checks and displays which obj_IDs appear multiple times in the table.
    
    Parameters
    ----------
    table : astropy.table.Table
        Input table containing 'obj_ID' column
    """
    # Count how many times each obj_ID appears
    id_counts = Counter(table['obj_ID'])
    
    # Find objects that appear more than once
    duplicates = {obj_id: count for obj_id, count in id_counts.items() if count > 1}
    
    # Display results
    print("\nDuplicate Analysis:")
    print("-" * 40)
    
    if not duplicates:
        print("No duplicate obj_IDs found!")
        return
    
    print("Found duplicate obj_IDs:")
    for obj_id, count in duplicates.items():
        # Find the row indices for this obj_ID
        indices = [i for i, val in enumerate(table['obj_ID']) if val == obj_id]
        print(f"\nobj_ID: {obj_id}")
        print(f"Appears {count} times")
        print(f"Found in rows: {indices}")

    


combined_table = read_and_combine_cat_files(base_path, exposure_ids)
#133000 combined all cat
filtered_cat = filter_combined_cat(combined_table)
#21401 eliminate cat that don't meet criteria
matched_table, filtered_cross_matching_cat = cross_match_lensing_improved(filtered_cat, lens_samples)
#2444 cross matching cat with lens table
table = filter_and_apply_overlap(matched_table)
#1966 all good fit/ available sources

# Extract the obj_ID column, converting to a list
obj_ids = table['obj_ID'].data

# Get unique obj_IDs
unique_obj_ids = np.unique(obj_ids)

# Randomly select 500 different obj_IDs (or less if there are not enough)
num_to_select = min(num, len(unique_obj_ids))
selected_obj_ids = np.random.choice(unique_obj_ids, size=num_to_select, replace=False)

# If you want to create a new table with these selected obj_IDs
selected_rows = table[np.isin(obj_ids, selected_obj_ids)]

images = []
for row in selected_rows:
    lens_id =  row['lens_sample_rows']
    survey = NewMockSurvey(config_path="./", config_file='csst_setting.yaml')
    survey.load_ideal_lens_from_table(lens_samples)
    this_sim_obj = survey.sim_obj_from(lens_id)
    survey.lensing_image_from(this_sim_obj)
    id_chip = row['ID_chip']
    band = 'g' if id_chip in [8, 23] else 'i' if id_chip in [7, 24] else 'r' if id_chip in [9, 22] else 'z'
    lensed_source_image = (this_sim_obj[0][band].blurred_image - this_sim_obj[0][band].blurred_lens_image)
    lensed_source_image = np.clip(lensed_source_image, 0, None)
    gal_img = galsim.InterpolatedImage(galsim.ImageF(lensed_source_image,scale=0.0185))
    # lensed_source_image= gal_img.rotate(theta=(PA-90)*galsim.degrees).drawImage(nx=128,ny=128,scale=0.037).array
    lensed_source_image= gal_img.drawImage(nx=128,ny=128,scale=0.037).array
    if np.sum(lensed_source_image) > 0:
        lensed_source_image /= np.sum(lensed_source_image)

    images.append(lensed_source_image)
# Save the table and images to a FITS file
fits_filename =  f'selected_target_{num}.fits'
primary_hdu = fits.PrimaryHDU(data=np.zeros((1, 1)))
primary_hdu.header['EXTEND'] = True

hdu1 = fits.table_to_hdu(selected_rows)
hdu2 = fits.ImageHDU(data=np.array(images), name='Lensed_Source_Images')

hdu_list = fits.HDUList([primary_hdu, hdu1, hdu2])
hdu_list.writeto(fits_filename, overwrite=True)

print(f"FITS file created: {fits_filename}")