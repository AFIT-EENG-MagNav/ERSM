import sys
import os
from copy import deepcopy
from os.path import dirname, realpath, exists
from struct import unpack

import pandas as pd
import numpy as np
import scipy.linalg as la
from scipy.spatial import distance
from scipy import interpolate
from tqdm import tqdm

sys.path.append(dirname(realpath(__file__)))
sys.path.append(dirname(dirname(realpath(__file__))))

import parseASPN as paspn
from Utils import coordinateUtils as cu
from Utils import Filters as filt
from Utils import ProcessingUtils as pu


SYNC_WORD = b'\xed\xa1\xda\x01'

LCM_HEADER_FORMAT = '>IQQII'
LCM_HEADER_LEN    = 28


def reindex(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.RangeIndex(len(df.index))
    return df

def add_epoch_col(df: pd.DataFrame) -> pd.DataFrame:
    df['epoch_sec'] = df.sec + (df.nsec * 1e-9)
    return df

def rm_dup_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    return reindex(df.drop_duplicates(['epoch_sec'], keep='last'))

def add_line_cols(log_df: pd.DataFrame,
                  wpt_df: pd.DataFrame):
    '''
    Add columns denoting line type and line number (if applicable)
    
    Parameters
    ----------
    log_df
        Dataframe compiled from flight log data
    wpt_df
        Dataframe of survey waypoints for each line. Columns include:
        
        - LINE_TYPE
            Type of line (0 for none, 1 for flight line, and 2 for tie line)
        - LINE
            Line number
        - START_LONG
            Line start longitude (dd)
        - START_LAT
            Line start latitude (dd)
        - END_LONG
            Line end longitude (dd)
        - END_LAT
            Line end latitude (dd)

    Returns
    -------
    log_df
        Log dataframe with added 'LINE_TYPE' and 'LINE' columns based
        on survey waypoints
    '''
    
    log_df['LINE_TYPE'] = 0
    log_df['LINE']      = 0

    log_coords = np.hstack([np.array(log_df.LONG)[:, np.newaxis],
                            np.array(log_df.LAT)[:, np.newaxis]])

    for _, row in wpt_df.iterrows():
        start_idx = distance.cdist(log_coords,
                                [[row.START_LONG, row.START_LAT]],
                                'euclidean').argmin()
        end_idx = distance.cdist(log_coords,
                                [[row.END_LONG, row.END_LAT]],
                                'euclidean').argmin()
        
        log_df.LINE_TYPE[start_idx:end_idx] = int(row.LINE_TYPE)
        log_df.LINE[start_idx:end_idx]      = int(row.LINE)
    
    return log_df

def convert_log(fname:       str,
                flight_name: str=None,
                lever_arm:   np.ndarray=np.zeros(3),
                rej_thresh:  float=500,
                interp_fs:   float=100,
                filt_cutoff: float=None,
                wpt_fname:   str=None) -> pd.DataFrame:
    '''
    Convert an LCM flight log into a MAMMAL compatible dataframe and
    save as a CSV (optional)
    
    Parameters
    ----------
    fname
        File path/name to the LCM flight log file
    flight_name
        File path/name for the saved MAMMAL compatible dataframe
        as a CSV. Set to None to skip this feature
    lever_arm
        1x3 array that specifies the x, y, z body-frame distances
        from the GNSS receiver antenna to the scalar
        magnetometer -> [x distance (m), y distance (m), z distance (m)]
    rej_thresh
        Core field rejection threshold (nT). Scalar samples outside the range
        determined by the calculated IGRF field +- this threshold will
        be removed from the dataset. For instance, if this threshold is
        500nT and the IGRF field is 50000nT, a scalar sample of 7000nT
        would be rejected, where a sample of 50300nT would not.
    interp_fs
        Sample/interpolation frequency of the output data
    filt_cutoff
        Low pass filter cutoff frequency for scalar data. Set to None
        to prevent filtering. **Note, this filtering is only applied to the
        'F' column data**
    wpt_fname
        File path/name to the csv holding survey flight and tie line
        waypoints
    
    Returns
    -------
    compiled_df
        MAMMAL compatible dataframe compiled from flight log data
    '''
    
    channel_list = ['SDM://CUB_0/PIXHAWK/1002.0/ALTITUDE',
                    'SDM://CUB_0/PIXHAWK/1010.0/EKF_PVA',
                    'aspn://v2.2/agilepod/twinleaf-vector-mag/barometricpressure',
                    'aspn://v2.2/agilepod/twinleaf-vector-mag/temperature',
                    'aspn://v2.2/agilepod/twinleaf-vector-mag/threeaxismagnetometer',
                    'ASPN://SYSTEM_NAME/MFAM/2009.0/MAG_HEAD_0',
                    'ASPN://SYSTEM_NAME/MFAM/2009.0/MAG_HEAD_1',
                    'aspn://OSPREY/MFAM/2008.0/VECTOR_MAG']
    
    print('Parsing LCM log:', fname)
    
    lcm_reader = LCM(fname)
    data_dict  = lcm_reader.parse_log(channel_list)
    
    alt_df = data_dict['SDM://CUB_0/PIXHAWK/1002.0/ALTITUDE']
    alt_df = add_epoch_col(alt_df)
    alt_df = rm_dup_timestamps(alt_df)
                    
    alt_ts = np.array(alt_df.epoch_sec)
    alt    = np.array(alt_df.altitude) # m
                    
    interp_alt = interpolate.interp1d(alt_ts, alt, 'linear', fill_value='extrapolate')

    ekf_df = data_dict['SDM://CUB_0/PIXHAWK/1010.0/EKF_PVA']
    ekf_df = add_epoch_col(ekf_df)
    ekf_df = rm_dup_timestamps(ekf_df)
    
    ekf_ts = np.array(ekf_df.epoch_sec)
    lat    = np.degrees(np.array(ekf_df['latitude']))  # rad2degrees
    lon    = np.degrees(np.array(ekf_df['longitude'])) # rad2degrees
    att_x  = np.degrees(np.array(ekf_df['x']))         # rad?
    att_y  = np.degrees(np.array(ekf_df['y']))         # rad?
    att_z  = np.degrees(np.array(ekf_df['z']))         # rad?
    
    interp_lat   = interpolate.interp1d(ekf_ts, lat,   'cubic',  fill_value='extrapolate')
    interp_lon   = interpolate.interp1d(ekf_ts, lon,   'cubic',  fill_value='extrapolate')
    interp_att_x = interpolate.interp1d(ekf_ts, att_x, 'linear', fill_value='extrapolate')
    interp_att_y = interpolate.interp1d(ekf_ts, att_y, 'linear', fill_value='extrapolate')
    interp_att_z = interpolate.interp1d(ekf_ts, att_z, 'linear', fill_value='extrapolate')

    baro_df = data_dict['aspn://v2.2/agilepod/twinleaf-vector-mag/barometricpressure']
    baro_df = add_epoch_col(baro_df)
    baro_df = rm_dup_timestamps(baro_df)
                    
    baro_ts = np.array(baro_df.epoch_sec)
    baro    = np.array(baro_df.pressure) # Pa
                    
    interp_baro = interpolate.interp1d(baro_ts, baro, 'linear', fill_value='extrapolate')

    temp_df = data_dict['aspn://v2.2/agilepod/twinleaf-vector-mag/temperature']
    temp_df = add_epoch_col(temp_df)
    temp_df = rm_dup_timestamps(temp_df)
                    
    temp_ts = np.array(temp_df.epoch_sec)
    temp    = np.array(temp_df.temperature) # C?
                    
    interp_temp = interpolate.interp1d(temp_ts, temp, 'linear', fill_value='extrapolate')
    
    vector_valid      = False
    mfam_vector_valid = False

    try:
        vector_df = data_dict['aspn://v2.2/agilepod/twinleaf-vector-mag/threeaxismagnetometer']
        vector_df = add_epoch_col(vector_df)
        vector_df = rm_dup_timestamps(vector_df)
        
        vector_ts = np.array(vector_df.epoch_sec)
        vector_x  = np.array(vector_df['X']) # nT
        vector_y  = np.array(vector_df['Y']) # nT
        vector_z  = np.array(vector_df['Z']) # nT
        vector_F  = la.norm([vector_df.X, vector_df.Y, vector_df.Z], axis=0)
        
        interp_vector_x = interpolate.interp1d(vector_ts, vector_x, 'linear', fill_value='extrapolate')
        interp_vector_y = interpolate.interp1d(vector_ts, vector_y, 'linear', fill_value='extrapolate')
        interp_vector_z = interpolate.interp1d(vector_ts, vector_z, 'linear', fill_value='extrapolate')
        
        vector_valid = True
    
    except KeyError:
        print('WARNING: Vector mag data not found!!!')

    try:
        # Pull data for MFAM sensor head 1
        scalar_1_df = data_dict['aspn://OSPREY/MFAM/2009.0/SCALAR_MAG_1']
        scalar_1_df = add_epoch_col(scalar_1_df)
        scalar_1_df = rm_dup_timestamps(scalar_1_df)
        scalar_1_ts = np.array(scalar_1_df.epoch_sec)

        # Pull data for MFAM sensor head 2
        scalar_2_df = data_dict['aspn://OSPREY/MFAM/2009.0/SCALAR_MAG_2']
        scalar_2_df = add_epoch_col(scalar_2_df)
        scalar_2_df = rm_dup_timestamps(scalar_2_df)
        scalar_2_ts = np.array(scalar_2_df.epoch_sec)
        
        try:
            # Pull data for MFAM vector sensor
            mfam_vector_df = data_dict['aspn://OSPREY/MFAM/2008.0/VECTOR_MAG']
            mfam_vector_df = add_epoch_col(mfam_vector_df)
            mfam_vector_df = rm_dup_timestamps(mfam_vector_df)
            
            mfam_vector_ts = np.array(mfam_vector_df.epoch_sec)
            mfam_vector_x  = np.array(mfam_vector_df['X']) # nT
            mfam_vector_y  = np.array(mfam_vector_df['Y']) # nT
            mfam_vector_z  = np.array(mfam_vector_df['Z']) # nT
            
            interp_mfam_vector_x = interpolate.interp1d( mfam_vector_ts, mfam_vector_x, 'linear', fill_value='extrapolate')
            interp_mfam_vector_y = interpolate.interp1d( mfam_vector_ts, mfam_vector_y, 'linear', fill_value='extrapolate')
            interp_mfam_vector_z = interpolate.interp1d( mfam_vector_ts, mfam_vector_z, 'linear', fill_value='extrapolate')
            
            mfam_vector_valid = True
        
        except KeyError:
            print('WARNING: MFAM vector mag data not found!!!')
        
        # Time-align data from both MFAM sensor heads
        min_t = max([scalar_1_ts.min(), scalar_2_ts.min()])
        max_t = min([scalar_1_ts.max(), scalar_2_ts.max()])
        
        scalar_1_df = scalar_1_df[scalar_1_df.epoch_sec >= min_t]
        scalar_1_df = scalar_1_df[scalar_1_df.epoch_sec <= max_t]
        
        scalar_2_df = scalar_2_df[scalar_2_df.epoch_sec >= min_t]
        scalar_2_df = scalar_2_df[scalar_2_df.epoch_sec <= max_t]
        
        if len(scalar_1_df) > len(scalar_2_df):
            scalar_1_df = scalar_1_df[:len(scalar_2_df) - len(scalar_1_df)]
        elif len(scalar_1_df) < len(scalar_2_df):
            scalar_2_df = scalar_2_df[:len(scalar_1_df) - len(scalar_2_df)]
        
        scalar_1_df = reindex(scalar_1_df)
        scalar_1_ts = np.array(scalar_1_df.epoch_sec)
        scalar_1_df['datetime'] = pd.to_datetime(scalar_1_df.epoch_sec, unit='s')
        
        scalar_2_df = reindex(scalar_2_df)
        scalar_2_ts = np.array(scalar_2_df.epoch_sec)
        scalar_2_df['datetime'] = pd.to_datetime(scalar_2_df.epoch_sec, unit='s')
        
        # Add approximate LLA data to calculate IGRF
        scalar_1_df['LAT']  = interp_lat(scalar_1_ts)
        scalar_1_df['LONG'] = interp_lon(scalar_1_ts)
        scalar_1_df['ALT']  = interp_alt(scalar_1_ts)
        
        scalar_2_df['LAT']  = interp_lat(scalar_2_ts)
        scalar_2_df['LONG'] = interp_lon(scalar_2_ts)
        scalar_2_df['ALT']  = interp_alt(scalar_2_ts)
        
        # NaN-out all extrapolated position data (in case the GPS was lost
        # at the start/end of the collect)
        min_t = max([ekf_ts.min(), alt_ts.min()])
        max_t = min([ekf_ts.max(), alt_ts.max()])
        
        scalar_1_pos_valid_mask = (scalar_1_df.epoch_sec < min_t) | (scalar_1_df.epoch_sec > max_t)
        scalar_2_pos_valid_mask = (scalar_2_df.epoch_sec < min_t) | (scalar_2_df.epoch_sec > max_t)
        
        scalar_1_df.LAT[scalar_1_pos_valid_mask]  = np.nan
        scalar_1_df.LONG[scalar_1_pos_valid_mask] = np.nan
        scalar_1_df.ALT[scalar_1_pos_valid_mask]  = np.nan
        
        scalar_2_df.LAT[scalar_2_pos_valid_mask]  = np.nan
        scalar_2_df.LONG[scalar_2_pos_valid_mask] = np.nan
        scalar_2_df.ALT[scalar_2_pos_valid_mask]  = np.nan
        
        # Use IGRF field to find when each sensor head values are valid
        scalar_1_df = pu.add_igrf_cols(scalar_1_df)
        scalar_2_df = pu.add_igrf_cols(scalar_2_df)
        
        min_F                 = scalar_1_df.IGRF_F - rej_thresh
        max_F                 = scalar_1_df.IGRF_F + rej_thresh
        scalar_1_valid_mask   = (scalar_1_df.magnitude >= min_F) & (scalar_1_df.magnitude <= max_F)
        interp_scalar_1_valid = interpolate.interp1d(scalar_1_df.epoch_sec,
                                                     scalar_1_valid_mask,
                                                     'linear',
                                                     fill_value='extrapolate')
        
        min_F                 = scalar_2_df.IGRF_F - rej_thresh
        max_F                 = scalar_2_df.IGRF_F + rej_thresh
        scalar_2_valid_mask   = (scalar_2_df.magnitude >= min_F) & (scalar_2_df.magnitude <= max_F)
        interp_scalar_2_valid = interpolate.interp1d(scalar_2_df.epoch_sec,
                                                     scalar_2_valid_mask,
                                                     'linear',
                                                     fill_value='extrapolate')
        
        # Find when both and neither sensor head values are valid
        both_valid_mask    = scalar_1_valid_mask & scalar_2_valid_mask
        neither_valid_mask = ~(scalar_1_valid_mask | scalar_2_valid_mask)
        
        # Test if all data is "bad"
        if neither_valid_mask.all():
            both_valid_mask    = ~both_valid_mask
            neither_valid_mask = ~neither_valid_mask
            
            print('WARNING: All scalar data was found to be outside the acceptable range from the expected IGRF magnitude, no data will be clipped!')
        
        # Find when only one sensor head is valid
        only_scalar_1_valid_mask = scalar_1_valid_mask & (~scalar_2_valid_mask)
        only_scalar_2_valid_mask = scalar_2_valid_mask & (~scalar_1_valid_mask)
        
        # Combine data from sensor heads
        scalar_combined = deepcopy(scalar_1_df)
        
        # Interpolate and filter valid sensor head data
        interp_scalar_1_lpf = interpolate.interp1d(scalar_1_df.epoch_sec[scalar_1_valid_mask],
                                                   scalar_1_df.magnitude[scalar_1_valid_mask],
                                                   'linear',
                                                   fill_value='extrapolate')
        interp_scalar_2_lpf = interpolate.interp1d(scalar_2_df.epoch_sec[scalar_2_valid_mask],
                                                   scalar_2_df.magnitude[scalar_2_valid_mask],
                                                   'linear',
                                                   fill_value='extrapolate')
        scalar_1_lpf = interp_scalar_1_lpf(scalar_1_df.epoch_sec)
        scalar_2_lpf = interp_scalar_2_lpf(scalar_2_df.epoch_sec)
        
        if filt_cutoff is not None:
            scalar_1_lpf = filt.lpf(scalar_1_lpf, filt_cutoff, interp_fs)
            scalar_2_lpf = filt.lpf(scalar_2_lpf, filt_cutoff, interp_fs)
        
        # Use filtered sensor head 1 when only sensor head 1 is valid
        scalar_combined.magnitude.iloc[only_scalar_1_valid_mask] = scalar_1_lpf[only_scalar_1_valid_mask]
        
        # Use filtered sensor head 2 when only sensor head 2 is valid
        scalar_combined.magnitude.iloc[only_scalar_2_valid_mask] = scalar_2_lpf[only_scalar_2_valid_mask]
        
        # Average sensor values when both are valid
        scalar_combined.magnitude.iloc[both_valid_mask] = (scalar_1_lpf[both_valid_mask] + scalar_2_lpf[both_valid_mask]) / 2.0
        
        # Drop when both heads are invalid
        scalar_combined.drop(neither_valid_mask.index[neither_valid_mask], inplace=True)
        
        # Interpolate and filter
        f = interpolate.interp1d(scalar_combined.epoch_sec,
                                 scalar_combined.magnitude,
                                 'linear',
                                 fill_value='extrapolate')
        
        min_t = scalar_combined.epoch_sec.min()
        max_t = scalar_combined.epoch_sec.max()
        scalar_combined_ts = np.linspace(min_t, max_t, int((max_t - min_t) * interp_fs))
        scalar_combined_F  = f(scalar_combined_ts)
        
        # Reinterpolate sensor head data at the new "combined" timestamps
        scalar_1_lpf = interp_scalar_1_lpf(scalar_combined_ts)
        scalar_2_lpf = interp_scalar_2_lpf(scalar_combined_ts)
        
        # Filter if cutoff freq is given
        if filt_cutoff is not None:
            scalar_combined_F = filt.lpf(scalar_combined_F, filt_cutoff, interp_fs)
            
            # Refilter individual sensor head data since these datasets were reinterpolated
            scalar_1_lpf = filt.lpf(scalar_1_lpf, filt_cutoff, interp_fs)
            scalar_2_lpf = filt.lpf(scalar_2_lpf, filt_cutoff, interp_fs)
        
        scalar_valid = True
    
    except KeyError:
        print('WARNING: Scalar mag data not found!!!')
        scalar_valid = False
    
    if vector_valid or scalar_valid:
        compiled_df = pd.DataFrame()
        
        if scalar_valid:
            # Add column of fused sensor head magnitude values
            compiled_ts      = scalar_combined_ts
            compiled_df['F'] = scalar_combined_F
            
            # Add raw columns for each sensor head
            interp_scalar_1 = interpolate.interp1d(scalar_1_df.epoch_sec,
                                                   scalar_1_df.magnitude,
                                                   'linear',
                                                   fill_value='extrapolate')
            interp_scalar_2 = interpolate.interp1d(scalar_2_df.epoch_sec,
                                                   scalar_2_df.magnitude,
                                                   'linear',
                                                   fill_value='extrapolate')
            
            compiled_df['SCALAR_1'] = interp_scalar_1(scalar_combined_ts)
            compiled_df['SCALAR_2'] = interp_scalar_2(scalar_combined_ts)
            
            # Add filtered columns for each sensor head
            compiled_df['SCALAR_1_LPF'] = scalar_1_lpf
            compiled_df['SCALAR_2_LPF'] = scalar_2_lpf
            
            # Add columns to specify when each sensor head had valid data
            compiled_scalar_1_valid = np.array(interp_scalar_1_valid(scalar_combined_ts))
            compiled_scalar_2_valid = np.array(interp_scalar_2_valid(scalar_combined_ts))
            
            # Any valid flag < 1 should be clamped to zero because of interpolation issues
            compiled_scalar_1_valid[compiled_scalar_1_valid != 1] = 0
            compiled_scalar_2_valid[compiled_scalar_2_valid != 1] = 0
            
            # Save valid flags as bool columns
            compiled_df['SCALAR_1_VALID'] = compiled_scalar_1_valid
            compiled_df['SCALAR_2_VALID'] = compiled_scalar_2_valid
            
            compiled_df['SCALAR_1_VALID'] = compiled_df['SCALAR_1_VALID'].astype('bool')
            compiled_df['SCALAR_2_VALID'] = compiled_df['SCALAR_2_VALID'].astype('bool')
        
        else:
            # Add column of scalar values from magnitude of VMR readings
            min_t = vector_ts.min()
            max_t = vector_ts.max()
            compiled_ts = np.linspace(min_t, max_t, int((max_t - min_t) * interp_fs))
            
            if filt_cutoff is not None:
                vector_F = filt.lpf(vector_F, filt_cutoff, interp_fs)
            
            compiled_df['F'] = vector_F
        
        # Add timestamp columns
        compiled_df['epoch_sec'] = compiled_ts
        compiled_df['datetime']  = pd.to_datetime(compiled_df['epoch_sec'], unit='s')
        
        # Apply lever arm to navigation LLA data to find true magnetometer LLA
        nav_lats = interp_lat(compiled_ts)
        nav_lons = interp_lon(compiled_ts)
        nav_alts = interp_alt(compiled_ts)
        
        roll  = interp_att_x(compiled_ts)
        pitch = interp_att_y(compiled_ts)
        yaw   = interp_att_z(compiled_ts)
        
        eulers = np.hstack([roll[:, np.newaxis],
                            pitch[:, np.newaxis],
                            yaw[:, np.newaxis]])
        
        lats = nav_lats.copy()
        lons = nav_lons.copy()
        alts = nav_alts.copy()
        
        if ~(lever_arm == np.zeros(3)).all():
            dcms = pu.angle2dcm(eulers,
                                angle_unit='degrees',
                                NED_to_body=False,
                                rotation_sequence=321)
            offsets = dcms @ lever_arm
            
            n = offsets[:, 0]
            e = offsets[:, 1]
            d = offsets[:, 2]
            
            lats = cu.coord_coord(nav_lats, nav_lons, n / 1000, np.zeros(len(n)))[:, 0]
            lons = cu.coord_coord(nav_lats, nav_lons, e / 1000, np.ones(len(n)) * 90)[:, 1]
            alts = nav_alts + d
        
        # Save interpolated logged and derrived data in a DataFrame
        compiled_df['NAV_LAT']  = nav_lats
        compiled_df['NAV_LONG'] = nav_lons
        compiled_df['NAV_ALT']  = nav_alts
        compiled_df['LAT']      = lats
        compiled_df['LONG']     = lons
        compiled_df['ALT']      = alts
        
        if vector_valid:
            compiled_df['X'] = interp_vector_x(compiled_ts)
            compiled_df['Y'] = interp_vector_y(compiled_ts)
            compiled_df['Z'] = interp_vector_z(compiled_ts)
        
        if mfam_vector_valid:
            compiled_df['X_MFAM'] = interp_mfam_vector_x(compiled_ts)
            compiled_df['Y_MFAM'] = interp_mfam_vector_y(compiled_ts)
            compiled_df['Z_MFAM'] = interp_mfam_vector_z(compiled_ts)
        
        compiled_df['ROLL']      = roll
        compiled_df['PITCH']     = pitch
        compiled_df['AZIMUTH']   = yaw
        compiled_df['BARO']      = interp_baro(compiled_ts)
        compiled_df['TEMP']      = interp_temp(compiled_ts)
        compiled_df['LINE']      = 0 # Placeholder values to be changed by 'add_line_cols()' when wpt_fname is given
        compiled_df['LINE_TYPE'] = 0 # Placeholder values to be changed by 'add_line_cols()' when wpt_fname is given
        
        # NaN-out all extrapolated position data (in case the GPS was lost
        # at the start/end of the collect)
        min_t = max([ekf_ts.min(), alt_ts.min()])
        max_t = min([ekf_ts.max(), alt_ts.max()])
        
        pos_valid_mask = (compiled_df.epoch_sec < min_t) | (compiled_df.epoch_sec > max_t)
        
        compiled_df.LAT[pos_valid_mask]  = np.nan
        compiled_df.LONG[pos_valid_mask] = np.nan
        compiled_df.ALT[pos_valid_mask]  = np.nan
        
        # Add line and line type data if wpt file given
        if wpt_fname is not None:
            compiled_df = add_line_cols(compiled_df,
                                        pd.read_csv(wpt_fname))
        
        if flight_name is not None:
            # Save the compiled log
            if exists(flight_name):
                raise Exception('{} already exists - file not overwritten'.format(flight_name))
            else:
                compiled_df.to_csv(flight_name, index=False)
    
    return compiled_df

def convert_ground_log(fname:       str,
		                lat:	   float,
		                lon:	   float,
		                alt:	   float,
		                rej_thresh:  float=6000,
                        interp_fs:   float=100,
                        filt_cutoff: float=None,
                        mag_1_channel: str='ASPN://Home/MFAM/2009.0/MAG_HEAD_0_filtered',
                        mag_2_channel: str='ASPN://Home/MFAM/2009.0/MAG_HEAD_1_filtered',
                        compass_channel: str='ASPN://Home/MFAM/2008.0/COMPASS_filtered'):

    '''
    Convert an LCM flight log into a MAMMAL compatible dataframe and
    save as a CSV (optional)
    
    Parameters
    ----------
    fname
        File path/name to the LCM ground station log file
    lat
        Latitude (dd) of ground station
    lon 
        Longitude (dd) of ground station
    alt 
        Altitude (dd) of ground station
    rej_thresh
        Core field rejection threshold (nT). Scalar samples outside the range
        determined by the calculated IGRF field +- this threshold will
        be removed from the dataset. For instance, if this threshold is
        500nT and the IGRF field is 50000nT, a scalar sample of 7000nT
        would be rejected, where a sample of 50300nT would not.
    interp_fs
        Sample/interpolation frequency of the output data
    filt_cutoff
        Low pass filter cutoff frequency for scalar data. Set to None
        to prevent filtering. **Note, this filtering is only applied to the
        'F' column data**
    mag_1_channel
        Channel name from one of the MFAM magnetomoeter sensors
    mag_2_channel
        Channel name from the other MFAM magnetometer sensors
    compass_channel
        Vector data from magnetometer     
    
    
    Returns
    -------
    compiled_df
        MAMMAL compatible dataframe compiled from ground station log data
        Dataframe - matches format from parseIM
        - DATE:      Date object (UTC)
        - TIME:      Number of seconds past UTC midnight
        - DOY:       Julian day of year (UTC)
        - X:         Magnetic field measurement in the North direction (nT)
        - Y:         Magnetic field measurement in the East direction (nT)
        - Z:         Magnetic field measurement in the Down direction (nT)
        - F:         Magnetic field measurement magnitude (nT)
        - datetime:  Datetime object (UTC)
        - epoch_sec: UNIX epoch timestamp (s)
        - LAT:       Latitude (dd)
        - LONG:      Longitude (dd)
        - ALT:       Altitude MSL (km)
        - IGRF_X:    IGRF magnetic field in the North direction (nT)
        - IGRF_Y:    IGRF magnetic field in the East direction (nT)
        - IGRF_Z:    IGRF magnetic field in the Down direction (nT)
        - IGRF_F:    IGRF magnetic field magnitude (nT)
    '''
    
    # Set Channel List	
    channel_list = [mag_1_channel, mag_2_channel, compass_channel]
    
    # Parse LCM packets
    print('Parsing LCM log:', fname)    
    lcm_reader = LCM(fname)
    data_dict  = lcm_reader.parse_log(channel_list)
    
    try:
        # Pull data for MFAM sensor head 1
        scalar_1_df = data_dict[mag_1_channel]
        scalar_1_df = add_epoch_col(scalar_1_df)
        scalar_1_df = rm_dup_timestamps(scalar_1_df)
        scalar_1_ts = np.array(scalar_1_df.epoch_sec)

        # Pull data for MFAM sensor head 2
        scalar_2_df = data_dict[mag_2_channel]
        scalar_2_df = add_epoch_col(scalar_2_df)
        scalar_2_df = rm_dup_timestamps(scalar_2_df)
        scalar_2_ts = np.array(scalar_2_df.epoch_sec)
        
        try:
            # Pull data for MFAM vector sensor
            mfam_vector_df = data_dict[compass_channel]
            mfam_vector_df = add_epoch_col(mfam_vector_df)
            mfam_vector_df = rm_dup_timestamps(mfam_vector_df)
            
            mfam_vector_ts = np.array(mfam_vector_df.epoch_sec)
            mfam_vector_x  = np.array(mfam_vector_df['X']) # nT
            mfam_vector_y  = np.array(mfam_vector_df['Y']) # nT
            mfam_vector_z  = np.array(mfam_vector_df['Z']) # nT
            
            interp_mfam_vector_x = interpolate.interp1d( mfam_vector_ts, mfam_vector_x, 'linear', fill_value='extrapolate')
            interp_mfam_vector_y = interpolate.interp1d( mfam_vector_ts, mfam_vector_y, 'linear', fill_value='extrapolate')
            interp_mfam_vector_z = interpolate.interp1d( mfam_vector_ts, mfam_vector_z, 'linear', fill_value='extrapolate')
            
            mfam_vector_valid = True
        
        except KeyError:
            print('WARNING: MFAM vector mag data not found!!!')
        
        # Time-align data from both MFAM sensor heads
        min_t = max([scalar_1_ts.min(), scalar_2_ts.min()])
        max_t = min([scalar_1_ts.max(), scalar_2_ts.max()])
        print(max_t-min_t)
        
        scalar_1_df = scalar_1_df[scalar_1_df.epoch_sec >= min_t]
        scalar_1_df = scalar_1_df[scalar_1_df.epoch_sec <= max_t]
        
        scalar_2_df = scalar_2_df[scalar_2_df.epoch_sec >= min_t]
        scalar_2_df = scalar_2_df[scalar_2_df.epoch_sec <= max_t]
        
        if len(scalar_1_df) > len(scalar_2_df):
            scalar_1_df = scalar_1_df[:len(scalar_2_df) - len(scalar_1_df)]
        elif len(scalar_1_df) < len(scalar_2_df):
            scalar_2_df = scalar_2_df[:len(scalar_1_df) - len(scalar_2_df)]
        
        scalar_1_df = reindex(scalar_1_df)
        scalar_1_ts = np.array(scalar_1_df.epoch_sec)
        scalar_1_df['datetime'] = pd.to_datetime(scalar_1_df.epoch_sec, unit='s')
        
        scalar_2_df = reindex(scalar_2_df)
        scalar_2_ts = np.array(scalar_2_df.epoch_sec)
        scalar_2_df['datetime'] = pd.to_datetime(scalar_2_df.epoch_sec, unit='s')
        
	
	# Add LLA values to scalar data frames (prior to IGRF calculation)
        scalar_1_df['LAT']  = lat
        scalar_1_df['LONG'] = lon
        scalar_1_df['ALT']  = alt
        
        scalar_2_df['LAT']  = lat
        scalar_2_df['LONG'] = lon
        scalar_2_df['ALT']  = alt
	        
        # Use IGRF field to find when each sensor head values are valid
        scalar_1_df = pu.add_igrf_cols(scalar_1_df)
        scalar_2_df = pu.add_igrf_cols(scalar_2_df)
        
        min_F                 = scalar_1_df.IGRF_F - rej_thresh
        max_F                 = scalar_1_df.IGRF_F + rej_thresh
        scalar_1_valid_mask   = (scalar_1_df.magnitude >= min_F) & (scalar_1_df.magnitude <= max_F)
        interp_scalar_1_valid = interpolate.interp1d(scalar_1_df.epoch_sec,
                                                     scalar_1_valid_mask,
                                                     'linear',
                                                     fill_value='extrapolate')
        
        min_F                 = scalar_2_df.IGRF_F - rej_thresh
        max_F                 = scalar_2_df.IGRF_F + rej_thresh
        scalar_2_valid_mask   = (scalar_2_df.magnitude >= min_F) & (scalar_2_df.magnitude <= max_F)
        interp_scalar_2_valid = interpolate.interp1d(scalar_2_df.epoch_sec,
                                                     scalar_2_valid_mask,
                                                     'linear',
                                                     fill_value='extrapolate')
        
        # Find when both and neither sensor head values are valid
        both_valid_mask    = scalar_1_valid_mask & scalar_2_valid_mask
        neither_valid_mask = ~(scalar_1_valid_mask | scalar_2_valid_mask)
        
        # Test if all data is "bad"
        if neither_valid_mask.all():
            both_valid_mask    = ~both_valid_mask
            neither_valid_mask = ~neither_valid_mask
            
            print('WARNING: All scalar data was found to be outside the acceptable range from the expected IGRF magnitude, no data will be clipped!')
        
        # Find when only one sensor head is valid
        only_scalar_1_valid_mask = scalar_1_valid_mask & (~scalar_2_valid_mask)
        only_scalar_2_valid_mask = scalar_2_valid_mask & (~scalar_1_valid_mask)
        
        # Combine data from sensor heads
        scalar_combined = deepcopy(scalar_1_df)
        
        # Interpolate and filter valid sensor head data
        print('Hello')
        print(scalar_1_df.epoch_sec[scalar_1_valid_mask])
        interp_scalar_1_lpf = interpolate.interp1d(scalar_1_df.epoch_sec[scalar_1_valid_mask],
                                                   scalar_1_df.magnitude[scalar_1_valid_mask],
                                                   'linear',
                                                   fill_value='extrapolate')
        print('Hello')
        print(scalar_2_df.epoch_sec[scalar_2_valid_mask])
        interp_scalar_2_lpf = interpolate.interp1d(scalar_2_df.epoch_sec[scalar_2_valid_mask],
                                                   scalar_2_df.magnitude[scalar_2_valid_mask],
                                                   'linear',
                                                   fill_value='extrapolate')
        scalar_1_lpf = interp_scalar_1_lpf(scalar_1_df.epoch_sec)
        scalar_2_lpf = interp_scalar_2_lpf(scalar_2_df.epoch_sec)
        
        if filt_cutoff is not None:
            scalar_1_lpf = filt.lpf(scalar_1_lpf, filt_cutoff, interp_fs)
            scalar_2_lpf = filt.lpf(scalar_2_lpf, filt_cutoff, interp_fs)
        
        # Use filtered sensor head 1 when only sensor head 1 is valid
        scalar_combined.magnitude.iloc[only_scalar_1_valid_mask] = scalar_1_lpf[only_scalar_1_valid_mask]
        
        # Use filtered sensor head 2 when only sensor head 2 is valid
        scalar_combined.magnitude.iloc[only_scalar_2_valid_mask] = scalar_2_lpf[only_scalar_2_valid_mask]
        
        # Average sensor values when both are valid
        scalar_combined.magnitude.iloc[both_valid_mask] = (scalar_1_lpf[both_valid_mask] + scalar_2_lpf[both_valid_mask]) / 2.0
        
        # Drop when both heads are invalid
        scalar_combined.drop(neither_valid_mask.index[neither_valid_mask], inplace=True)
        
        # Interpolate and filter
        f = interpolate.interp1d(scalar_combined.epoch_sec,
                                 scalar_combined.magnitude,
                                 'linear',
                                 fill_value='extrapolate')
        
        min_t = scalar_combined.epoch_sec.min()
        max_t = scalar_combined.epoch_sec.max()
        
        #epoch_sec
        scalar_combined_ts = np.linspace(min_t, max_t, int((max_t - min_t) * interp_fs))
        scalar_combined_F  = f(scalar_combined_ts)
        
        # Reinterpolate sensor head data at the new "combined" timestamps
        scalar_1_lpf = interp_scalar_1_lpf(scalar_combined_ts)
        scalar_2_lpf = interp_scalar_2_lpf(scalar_combined_ts)
        
        # Filter if cutoff freq is given
        if filt_cutoff is not None:
            scalar_combined_F = filt.lpf(scalar_combined_F, filt_cutoff, interp_fs)
            
            # Refilter individual sensor head data since these datasets were reinterpolated
            scalar_1_lpf = filt.lpf(scalar_1_lpf, filt_cutoff, interp_fs)
            scalar_2_lpf = filt.lpf(scalar_2_lpf, filt_cutoff, interp_fs)
        
        scalar_valid = True
    
    except KeyError:
        print('WARNING: Scalar mag data not found!!!')
        scalar_valid = False
        
    # TODO Compute and Return Data Frame
    compiled_df = pd.DataFrame()
    
    # Handle Timings
    compiled_df['epoch_sec'] = scalar_combined_ts
    compiled_df['datetime'] = pd.to_datetime(compiled_df['epoch_sec'], unit='s')
    compiled_df['DATE'] = compiled_df['datetime'].dt.date
    compiled_df['TIME'] = compiled_df['datetime'].dt.time
    compiled_df['DOY'] = compiled_df['datetime'].dt.day_of_year
        
    # Handle B-Field Readings
    compiled_df['X'] = interp_mfam_vector_x(scalar_combined_ts)
    compiled_df['Y'] = interp_mfam_vector_y(scalar_combined_ts)
    compiled_df['Z'] = interp_mfam_vector_z(scalar_combined_ts)
    compiled_df['F'] = scalar_combined_F
        
    # Handle Positioning and Offset
    compiled_df['LAT']  = lat
    compiled_df['LONG'] = lon
    compiled_df['ALT']  = alt
    compiled_df = pu.add_igrf_cols(compiled_df)
    return compiled_df

def convert_ground_log_acq(directory:       str,
		                    lat:	   float,
		                    lon:	   float,
		                    alt:	   float,
		                    rej_thresh:  float=50000,
                            interp_fs:   float=100,
                            filt_cutoff: float=None,
                            mag_1_channel: str='ASPN://Home/MFAM/2009.0/MAG_HEAD_0_filtered',
                            mag_2_channel: str='ASPN://Home/MFAM/2009.0/MAG_HEAD_1_filtered',
                            compass_channel: str='ASPN://Home/MFAM/2008.0/COMPASS_filtered'):
    '''
    Convert an LCM flight log into a MAMMAL compatible dataframe and
    save as a CSV (optional)
    
    Parameters
    ----------
    directory
        Name of directory containing all of the log files from ground_station
    lat
        Latitude (dd) of ground station
    lon 
        Longitude (dd) of ground station
    alt 
        Altitude (dd) of ground station
    rej_thresh
        Core field rejection threshold (nT). Scalar samples outside the range
        determined by the calculated IGRF field +- this threshold will
        be removed from the dataset. For instance, if this threshold is
        500nT and the IGRF field is 50000nT, a scalar sample of 7000nT
        would be rejected, where a sample of 50300nT would not.
    interp_fs
        Sample/interpolation frequency of the output data
    filt_cutoff
        Low pass filter cutoff frequency for scalar data. Set to None
        to prevent filtering. **Note, this filtering is only applied to the
        'F' column data**
    mag_1_channel
        Channel name from one of the MFAM magnetomoeter sensors
    mag_2_channel
        Channel name from the other MFAM magnetometer sensors
    compass_channel
        Vector data from magnetometer     
    
    
    Returns
    -------
    compiled_df
        MAMMAL compatible dataframe compiled from ground station log data
        Dataframe - matches format from parseIM
        - DATE:      Date object (UTC)
        - TIME:      Number of seconds past UTC midnight
        - DOY:       Julian day of year (UTC)
        - X:         Magnetic field measurement in the North direction (nT)
        - Y:         Magnetic field measurement in the East direction (nT)
        - Z:         Magnetic field measurement in the Down direction (nT)
        - F:         Magnetic field measurement magnitude (nT)
        - datetime:  Datetime object (UTC)
        - epoch_sec: UNIX epoch timestamp (s)
        - LAT:       Latitude (dd)
        - LONG:      Longitude (dd)
        - ALT:       Altitude MSL (km)
        - IGRF_X:    IGRF magnetic field in the North direction (nT)
        - IGRF_Y:    IGRF magnetic field in the East direction (nT)
        - IGRF_Z:    IGRF magnetic field in the Down direction (nT)
        - IGRF_F:    IGRF magnetic field magnitude (nT)
    '''
    
    log_df = pd.DataFrame()
    for _, _, files in os.walk(directory):
    	print(files)
    	for file_name in files:
    	    absolute_path = os.path.join(directory, file_name)
    	    log_df = pd.concat([log_df,convert_ground_log(absolute_path, lat, lon, alt, rej_thresh, interp_fs, filt_cutoff, mag_1_channel, mag_2_channel, compass_channel)]).sort_values(by=['DOY', 'TIME']).dropna()
    return rm_dup_timestamps(log_df)
	
class LCM():
    def __init__(self, fname=None):
        self.fname = fname
        
        self.offset = 0
        self.loaded = False
        
        self.reset()
        self.open()
    
    def reset(self):
        self.event_num   = None
        self.timestamp   = None
        self.channel_len = None
        self.data_len    = None
        self.data_bytes  = None
    
    def open(self, fname=None):
        if fname is not None:
            self.fname = fname
        
        self.reset()
        
        # Read the entire log file as a byte array
        with open(self.fname, 'rb') as log:
            self.contents = log.read()
        
        self.num_messages = self.contents.count(SYNC_WORD) - 1
        
        self.offset = -1
        self.loaded = True
    
    def data(self):
        return {'event_num':   self.event_num,
                'timestamp':   self.timestamp,
                'channel':     self.channel,
                'data_bytes':  self.data_bytes,
                'aspn_packet': self.aspn_packet}
    
    def read_next(self):
        if not self.loaded:
            self.open()
        
        self.reset()
        
        # Find next LCM message
        self.offset = self.contents.find(SYNC_WORD, self.offset + 1)
        
        # Return -1 if no more messages exist in log,
        # process header if another message is found
        if self.offset == -1:
            return -1
        
        else:
            _, self.event_num, self.timestamp, channel_len, data_len = unpack(LCM_HEADER_FORMAT, self.contents[self.offset:self.offset + LCM_HEADER_LEN])
            self.timestamp /= 1000000.0 # us to s
            
            channel_start = self.offset + LCM_HEADER_LEN
            channel_end   = channel_start + channel_len
            
            self.channel = str(self.contents[channel_start:channel_end], 'utf-8')
            
            data_start = channel_end
            data_end   = channel_end + data_len
            self.data_bytes  = self.contents[data_start:data_end]
            self.aspn_packet = paspn.parse_aspn(self.channel, self.data_bytes)
        
        return self.data()
    
    def parse_log(self, channel_list=None):
        all_ids = set()
        self.channels = []
        
        data_dict = {}
        
        for _ in tqdm(range(self.num_messages)):
            lcm_packet = self.read_next()
            
            lcm_event_num  = lcm_packet['event_num']
            lcm_timestamp  = lcm_packet['timestamp']
            lcm_channel    = lcm_packet['channel']
            lcm_data_bytes = lcm_packet['data_bytes']
            aspn_packet    = lcm_packet['aspn_packet']
            
            fingerprint    = aspn_packet['fingerprint']
            icd            = aspn_packet['icd']
            seq_num        = aspn_packet['seq_num']
            sec            = aspn_packet['sec']
            nsec           = aspn_packet['nsec']
            valid_sec      = aspn_packet['valid_sec']
            valid_nsec     = aspn_packet['valid_nsec']
            id             = aspn_packet['id']
            payload_bytes  = aspn_packet['payload_bytes']
            parsed_payload = aspn_packet['parsed_payload']
            all_ids.add(id)
            if lcm_channel not in self.channels:
                self.channels.append(lcm_channel)

            process = False
            
            if (channel_list is not None):
                if lcm_channel in channel_list:
                    process = True
            else:
                process = True
            
            if process:
                log_dict = {'event_num':   lcm_event_num, 
                            'timestamp':   lcm_timestamp, 
                            'channel':     lcm_channel, 
                            'fingerprint': fingerprint,
                            'icd':         icd,
                            'seq_num':     seq_num,
                            'sec':         sec,
                            'nsec':        nsec,
                            'valid_sec':   valid_sec,
                            'valid_nsec':  valid_nsec,
                            'id':          id}
                log_dict.update(parsed_payload)
                
                if 'covariance' in log_dict.keys():
                    log_dict['covariance'] = [log_dict['covariance']]
                
                if lcm_channel not in data_dict.keys():
                    data_dict[lcm_channel] = []
                    
                data_dict[lcm_channel].append(log_dict)
        
        for channel in data_dict.keys():
            data_dict[channel] = pd.DataFrame(data_dict[channel])
        print(all_ids)
        return data_dict


if __name__ == '__main__':
    import datetime as dt
    
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import pytz
    from matplotlib import cm
    
    
    LOG_FNAME = r'D:\AFIT\Research\WPAFB_Survey\data\flight_data\2022-08-03\flight4\project_20220803_18_03_58.00'
    CSV_FNAME = r'D:\AFIT\Research\WPAFB_Survey\data\flight_data\2022-08-03\flight4\project_20220803_18_03_58_converted_log.csv'
    
    SAVE_CSV = False
    PLOT     = True
    
    TITLE = 'Center Survey Raw Magnitude\n(20220803_18_03_58)'
    
    START_UTC = dt.datetime(2022, 8, 3, 18, 37, 51, tzinfo=pytz.utc)
    START_UTC = dt.datetime.fromtimestamp(START_UTC.timestamp())
    
    END_UTC = dt.datetime(2022, 8, 3, 18, 49, 40, tzinfo=pytz.utc)
    END_UTC = dt.datetime.fromtimestamp(END_UTC.timestamp())
    
    
    img = mpimg.imread(r'D:\AFIT\Research\WPAFB_Survey\data\wpafb.jpg')
    
    log_df    = convert_log(LOG_FNAME)
    time_mask = (log_df.epoch_sec >= START_UTC.timestamp()) & (log_df.epoch_sec <= END_UTC.timestamp())
    log_df    = log_df[time_mask]
    
    if SAVE_CSV:
        log_df.to_csv(CSV_FNAME, index=False)
    
    if PLOT:
        plt.figure()
        plt.title(TITLE)
        plt.xlabel('Longitude (dd)')
        plt.ylabel('Latitude (dd)')
        plt.imshow(img, extent=[-84.121190, -84.095010, 39.771541, 39.777271])
        plt.plot(log_df.LONG, log_df.LAT, label='Flight Path')
        cb = plt.scatter(log_df.LONG, log_df.LAT, s=60, c=log_df.F, cmap=cm.coolwarm)
        plt.colorbar(cb, label='nT')
        plt.legend()
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(TITLE)
        ax.set_xlabel('Longitude (dd)')
        ax.set_ylabel('Latitude (dd)')
        ax.set_zlabel('Altitude MSL (m)')
        ax.scatter(log_df.LONG, log_df.LAT, log_df.ALT, s=60, c=log_df.F, cmap=cm.coolwarm)
        plt.colorbar(cb, label='nT')
        
        plt.show(block=True)
