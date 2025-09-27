from io import BytesIO
from struct import unpack


# ASPN stuff
ASPN_HEADER_FORMAT = '>QbqqiqiI'
ASPN_HEADER_LEN    = 45

ACCUMULATED_DIST_TRAV_ICD      = '1301.0'
ALTITUDE_ICD                   = '1002.0'
ALTITUDE_1D_ICD                = '1006.1'
ALTITUDE_2D_ICD                = '1006.2'
ALTITUDE_3D_ICD                = '1006.3'
BAROMETRIC_PRESSURE_ICD        = '2011.0'
DELTA_POSITION_1D_ICD          = '1302.1'
DELTA_POSITION_2D_ICD          = '1302.2'
DELTA_POSITION_3D_ICD          = '1302.3'
DELTA_RANGE_ICD                = '1305.0'
DELTA_ROTATION_1D_ICD          = '1304.1'
DELTA_ROTATION_2D_ICD          = '1304.2'
DELTA_ROTATION_3D_ICD          = '1304.3'
DELTA_VELOCITY_1D_ICD          = '1303.1'
DELTA_VELOCITY_2D_ICD          = '1303.2'
DELTA_VELOCITY_3D_ICD          = '1303.3'
DIRECTION_OF_MOTION_2D_ICD     = '1306.2'
DIRECTION_OF_MOTION_3D_ICD     = '1306.3'
DISCRETE_EVENT_ICD             = '3001.0'
EVENT_LOG_ICD                  = '3003.0'
GEODETIC_POSITION_2D_ICD       = '1001.2'
GEODETIC_POSITION_3D_ICD       = '1001.3'
IMU_ICD                        = '2001.0'
INS_ICD                        = '2002.0'
SCALAR_MAGNETOMETER_ICD        = '2009.0'
PLATFORM_TYPE_ICD              = '3002.0'
POSITION_ATTITUDE_ICD          = '1011.0'
POSITION_VELOCITY_ICD          = '1012.0'
POSITION_VELOCITY_ATTITUDE_ICD = '1010.0'
SPEED_ICD                      = '1005.0'
TEMPERATURE_ICD                = '2012.0'
VECTOR_MAG_ICD                 = '2008.0'
VELOCITY_1D_ICD                = '1004.1'
VELOCITY_2D_ICD                = '1004.2'
VELOCITY_3D_ICD                = '1004.3'


def parse_aspn(channel, data_bytes):
    '''
    
    '''
    
    fingerprint, icd, seq_num, sec, nsec, valid_sec, valid_nsec, id_len = unpack(ASPN_HEADER_FORMAT, data_bytes[:ASPN_HEADER_LEN])
    id_start = ASPN_HEADER_LEN
    id_end   = id_start + id_len
    
    id = data_bytes[id_start:id_end].decode('utf-8', errors='ignore')
    payload_bytes  = data_bytes[id_end:]
    parsed_payload = parse_payload_bytes(channel, payload_bytes)

    return {'fingerprint':    fingerprint,
            'icd':            icd,
            'seq_num':        seq_num,
            'sec':            sec,
            'nsec':           nsec,
            'valid_sec':      valid_sec,
            'valid_nsec':     valid_nsec,
            'id':             id,
            'payload_bytes':  payload_bytes,
            'parsed_payload': parsed_payload}

def parse_payload_bytes(channel, payload_bytes):
    '''
    
    '''
    bio = BytesIO(payload_bytes)
    
    if ACCUMULATED_DIST_TRAV_ICD in channel:
        delta_t, distance, variance = unpack('>3d', bio.read(8 * 3))

        return {'delta_t':  delta_t,
                'distance': distance,
                'variance': variance}
    
    if ALTITUDE_ICD in channel:
        altitude, variance = unpack('>2d', bio.read(8 * 2))
        
        return {'altitude': altitude,
                'variance': variance}
    
    if ALTITUDE_1D_ICD in channel:
        x, variance = unpack('>2d', bio.read(8 * 2))
        
        return {'x':        x,
                'variance': variance}
    
    if ALTITUDE_2D_ICD in channel:
        x, y       = unpack('>2d', bio.read(8 * 2))
        covariance = unpack('>4d', bio.read(8 * 4))
        
        return {'x':          x,
                'y':          y,
                'covariance': covariance}
    
    if ALTITUDE_3D_ICD in channel:
        x, y, z    = unpack('>3d', bio.read(8 * 3))
        covariance = unpack('>9d', bio.read(8 * 9))
        
        return {'x':          x,
                'y':          y,
                'z':          z,
                'covariance': covariance}
    
    if (BAROMETRIC_PRESSURE_ICD in channel) or ('barometricpressure' in channel):
        pressure = unpack('>d', bio.read(8 * 1))[0]
        
        return {'pressure': pressure}
    
    if DELTA_POSITION_1D_ICD in channel:
        delta_t, x, variance = unpack('>3d', bio.read(8 * 3))

        return {'delta_t':  delta_t,
                'x':        x,
                'variance': variance}
    
    if DELTA_POSITION_2D_ICD in channel:
        delta_t, x, y = unpack('>3d', bio.read(8 * 3))
        covariance    = unpack('>4d', bio.read(8 * 4))
        
        return {'delta_t':    delta_t,
                'x':          x,
                'y':          y,
                'covariance': covariance}
    
    if DELTA_POSITION_3D_ICD in channel:
        try:
            delta_t, x, y, z = unpack('>4d', bio.read(8 * 4))
            covariance       = unpack('>9d', bio.read(8 * 9))
        except:
            delta_t    = None
            x          = None
            y          = None
            z          = None
            covariance = None
        
        return {'delta_t':    delta_t,
                'x':          x,
                'y':          y,
                'z':          z,
                'covariance': covariance}
    
    if DELTA_RANGE_ICD in channel:
        delta_t, delta_range, variance = unpack('>3d', bio.read(8 * 3))

        return {'delta_t':     delta_t,
                'delta_range': delta_range,
                'variance':    variance}
    
    if DELTA_ROTATION_1D_ICD in channel:
        delta_t, x, variance = unpack('>3d', bio.read(8 * 3))

        return {'delta_t':  delta_t,
                'x':        x,
                'variance': variance}
    
    if DELTA_ROTATION_2D_ICD in channel:
        delta_t, x, y = unpack('>3d', bio.read(8 * 3))
        covariance    = unpack('>4d', bio.read(8 * 4))

        return {'delta_t':    delta_t,
                'x':          x,
                'y':          y,
                'covariance': covariance}
    
    if DELTA_ROTATION_3D_ICD in channel:
        delta_t, x, y, z = unpack('>4d', bio.read(8 * 4))
        covariance       = unpack('>9d', bio.read(8 * 9))

        return {'delta_t':    delta_t,
                'x':          x,
                'y':          y,
                'z':          z,
                'covariance': covariance}
    
    if DELTA_VELOCITY_1D_ICD in channel:
        delta_t, x, variance = unpack('>3d', bio.read(8 * 3))

        return {'delta_t':  delta_t,
                'x':        x,
                'variance': variance}
    
    if DELTA_VELOCITY_2D_ICD in channel:
        delta_t, x, y = unpack('>3d', bio.read(8 * 3))
        covariance    = unpack('>4d', bio.read(8 * 4))

        return {'delta_t':    delta_t,
                'x':          x,
                'y':          y,
                'covariance': covariance}
    
    if DELTA_VELOCITY_3D_ICD in channel:
        delta_t, x, y, z = unpack('>4d', bio.read(8 * 4))
        covariance       = unpack('>9d', bio.read(8 * 9))

        return {'delta_t':    delta_t,
                'x':          x,
                'y':          y,
                'z':          z,
                'covariance': covariance}
    
    if DIRECTION_OF_MOTION_2D_ICD in channel:
        x, y       = unpack('>2d', bio.read(8 * 2))
        covariance = unpack('>4d', bio.read(8 * 4))

        return {'x':          x,
                'y':          y,
                'covariance': covariance}
    
    if DIRECTION_OF_MOTION_3D_ICD in channel:
        x, y, z    = unpack('>3d', bio.read(8 * 3))
        covariance = unpack('>9d', bio.read(8 * 9))

        return {'x':          x,
                'y':          y,
                'z':          z,
                'covariance': covariance}
    
    if DISCRETE_EVENT_ICD in channel:
        event_type, id_len = unpack('>bI', bio.read(1 + 4))
        device_id          = bio.read(id_len).decode('utf-8', errors='ignore')

        return {'event_type': event_type,
                'device_id':  device_id}
    
    
    try:
    	if EVENT_LOG_ICD in channel:
        	event_log_type, log_len = unpack('>bI', payload_bytes[1 + 4])
        	eventlog                = bio.read(log_len).decode('utf-8', errors='ignore')

        	return {'event_log_type': event_log_type,
                'eventlog':       eventlog}
    except:
    	print("Issue with 3003")
    
    if GEODETIC_POSITION_2D_ICD in channel:
        latitude, longitude = unpack('>2d', bio.read(8 * 2))
        covariance          = unpack('>4d', bio.read(8 * 4))

        return {'latitude':   latitude,
                'longitude':  longitude,
                'covariance': covariance}
    
    if GEODETIC_POSITION_3D_ICD in channel:
        try:
            latitude, longitude, altitude = unpack('>3d', bio.read(8 * 3))
            covariance                    = unpack('>9d', bio.read(8 * 9))
        except:
            latitude   = None
            longitude  = None
            altitude   = None
            covariance = None

        return {'latitude':   latitude,
                'longitude':  longitude,
                'altitude':   altitude,
                'covariance': covariance}
    
    if (IMU_ICD in channel) or (('imu' in channel) and (INS_ICD not in channel)):
        delta_v_0, delta_v_1, delta_v_2, delta_theta_0, delta_theta_1, delta_theta_2 = unpack('>6d', bio.read(8 * 6))

        return {'delta_v[0]':     delta_v_0,
                'delta_v[1]':     delta_v_1,
                'delta_v[2]':     delta_v_2,
                'delta_theta[0]': delta_theta_0,
                'delta_theta[1]': delta_theta_1,
                'delta_theta[2]': delta_theta_2}
    
    if INS_ICD in channel:
        latitude, longitude, altitude = unpack('>3d', bio.read(8 * 3))
        covariance                    = unpack('>9d', bio.read(8 * 9))
        delta_v_0, delta_v_1, delta_v_2, delta_theta_0, delta_theta_1, delta_theta_2 = unpack('>6d', bio.read(8 * 6))

        return {'latitude':       latitude,
                'longitude':      longitude,
                'altitude':       altitude,
                'covariance':     covariance,
                'delta_v[0]':     delta_v_0,
                'delta_v[1]':     delta_v_1,
                'delta_v[2]':     delta_v_2,
                'delta_theta[0]': delta_theta_0,
                'delta_theta[1]': delta_theta_1,
                'delta_theta[2]': delta_theta_2}
    
    if SCALAR_MAGNETOMETER_ICD in channel:
        magnitude, variance = unpack('>2d', bio.read(8 * 2))
        
        return {'magnitude': magnitude,
                'variance':  variance}
    
    if PLATFORM_TYPE_ICD in channel:
        platform_type = unpack('>d', bio.read(8 * 1))[0]
        
        return {'platform_type': platform_type}
    
    if POSITION_ATTITUDE_ICD in channel:
        latitude, longitude, altitude, x, y, z = unpack('>6d',  bio.read(8 * 6))
        covariance                             = unpack('>36d', bio.read(8 * 36))

        return {'latitude':   latitude,
                'longitude':  longitude,
                'altitude':   altitude,
                'x':          x,
                'y':          y,
                'z':          z,
                'covariance': covariance}
    
    if POSITION_VELOCITY_ICD in channel:
        latitude, longitude, altitude, velocity_0, velocity_1, velocity_2 = unpack('>6d',  bio.read(8 * 6))
        covariance = unpack('>36d', bio.read(8 * 36))

        return {'latitude':   latitude,
                'longitude':  longitude,
                'altitude':   altitude,
                'velocity_0': velocity_0,
                'velocity_1': velocity_1,
                'velocity_2': velocity_2,
                'covariance': covariance}
    
    if POSITION_VELOCITY_ATTITUDE_ICD in channel:
        latitude, longitude, altitude, velocity_0, velocity_1, velocity_2, x, y, z = unpack('>9d',  bio.read(8 * 9))
        covariance = unpack('>36d', bio.read(8 * 36))

        return {'latitude':   latitude,
                'longitude':  longitude,
                'altitude':   altitude,
                'velocity_0': velocity_0,
                'velocity_1': velocity_1,
                'velocity_2': velocity_2,
                'x':          x,
                'y':          y,
                'z':          z,
                'covariance': covariance}
    
    if SPEED_ICD in channel:
        speed, variance = unpack('>2d', bio.read(8 * 2))
        
        return {'speed':    speed,
                'variance': variance}
    
    if (TEMPERATURE_ICD in channel) or ('temperature' in channel):
        temperature = unpack('>d', bio.read(8 * 1))[0]
        
        return {'temperature': temperature}
    
    if (VECTOR_MAG_ICD in channel) or ('threeaxismagnetometer' in channel):
        X, Y, Z    = unpack('>3d', bio.read(8 * 3))
        covariance = unpack('>9d', bio.read(8 * 9))
        
        return {'X':          X,
                'Y':          Y,
                'Z':          Z,
                'covariance': covariance}
    
    if VELOCITY_1D_ICD in channel:
        x, variance = unpack('>2d', bio.read(8 * 2))
        
        return {'x':        x,
                'variance': variance}
    
    if VELOCITY_2D_ICD in channel:
        x, y       = unpack('>2d', bio.read(8 * 2))
        covariance = unpack('>4d', bio.read(8 * 4))

        return {'x':          x,
                'y':          y,
                'covariance': covariance}
    
    if VELOCITY_3D_ICD in channel:
        x, y, z    = unpack('>3d', bio.read(8 * 3))
        covariance = unpack('>9d', bio.read(8 * 9))

        return {'x':          x,
                'y':          y,
                'z':          z,
                'covariance': covariance}
    
    return {}
