"""CDM module.

This module loads CDMS either from a .pdf-file or from the SQL database.

Example
-------

Notes
-----

Attributes
----------

"""
import numpy as np
import pandas as pd
import fitz  # this is pymupdf
from datetime import datetime, timezone
import mysql.connector as sql
from sqlalchemy import create_engine
from astropy import units as u


class CDM:
    """
    A Conjunction Data Message.
    Further information: https://public.ccsds.org/Pubs/508x0b1e2c2.pdf
    """


    def __init__(self):
        """
        
        """
        # self.objectA = SatObject("ObjectA")
        # self.objectB = SatObject("ObjectB")
    

    def load_cdm_from_pdf(self, filename):
        """
        Loads a CDM from a .pdf file.
        """
        doc = fitz.open(filename)  # open document
        for page in doc:  # iterate the document pages
            text = str(page.get_text().encode("utf8"))  # get plain text (is in UTF-8)
            # TODO: add test for more than one pdf page
        
        lines = text.split('\\n')

        self.generation_date = datetime.strptime(lines[2].rpartition(": ")[2].replace(" ",""),'%Y/%m/%d%H:%M').replace(tzinfo=timezone.utc)
        # print(generation_date)

        self.ccsds_cdm_vers = lines[5].rpartition(":")[2].strip()
        # print(ccsds_cdm_vers)
        self.creation_date = datetime.strptime(lines[6].rpartition(": ")[2].replace(" ",""),'%Y/%m/%d%H:%M:%S').replace(tzinfo=timezone.utc)
        # print(creation_date)
        self.originator = lines[7].rpartition(":")[2].strip()
        # print(originator)
        self.message_for = lines[8].rpartition(":")[2].strip()
        # print(message_for)
        self.message_id = lines[9].rpartition(":")[2].strip()
        # print(message_id)

        self.tca = datetime.strptime(lines[11].split(": ")[1].replace(" ",""),'%Y/%m/%d%H:%M:%S.%f').replace(tzinfo=timezone.utc)
        # print(tca)
        self.miss_distance = float(lines[12].split(": ")[1].strip())
        # print(miss_distance)
        self.relative_speed = float(lines[13].split(": ")[1].strip())
        # print(relative_speed)
        rel_position_R = float(lines[14].split(": ")[1].strip())
        # print(relative_position_R)
        rel_position_T = float(lines[15].split(": ")[1].strip())
        # print(relative_position_T)
        rel_position_N = float(lines[16].split(": ")[1].strip())
        # print(relative_position_N)
        self.rel_position_RTN = np.array([[rel_position_R],[rel_position_T],[rel_position_N]])
        self.collision_prob = float(lines[17].split(": ")[1].strip())
        # print(collision_prob)
        self.collision_prob_method = lines[18].split(": ")[1].strip()
        # print(collision_prob_method)

        line = lines[22].split(": ")[1].strip().split("  ")
        self.object_designator = [x for x in line if x]
        # print(object_designator)
        line = lines[23].split(": ")[1].strip().split("  ")
        self.object_name = [x for x in line if x]
        # print(object_name)
        line = lines[24].split(": ")[1].strip().split("  ")
        self.itn_designator = [x for x in line if x]
        # print(itn_designator)
        line = lines[25].split(": ")[1].strip().split("  ")
        self.object_type = [x for x in line if x]
        # print(object_type)
        line = lines[26].split(": ")[1].strip().split("  ")
        self.operator_organization = [x for x in line if x]
        # print(operator_organization)
        line = lines[27].split(": ")[1].strip().split("  ")
        self.ephemeris_name = [x for x in line if x]
        # print(ephemeris_name)
        line = lines[28].split(": ")[1].strip().split("  ")
        self.maneuverable = [x for x in line if x]
        # print(maneuverable)
        line = lines[29].split(": ")[1].strip().split("  ")
        self.ref_frame = [x for x in line if x]
        # print(ref_frame)
        line = lines[30].split(" : ")[1].strip().split("  ")
        self.gravity_model = [x for x in line if x]
        # print(gravity_model)
        line = lines[31].split(": ")[1].strip().split("  ")
        self.atmospheric_model = [x for x in line if x]
        # print(atmospheric_model)
        line = lines[32].split(": ")[1].strip().split("  ")
        self.n_body_perturbations = [x for x in line if x]
        # print(n_body_perturbations)
        line = lines[33].split(": ")[1].strip().split("  ")
        self.solar_rad_pressure = [x for x in line if x]
        # print(solar_rad_pressure)
        line = lines[34].split(": ")[1].strip().split("  ")
        self.earth_tides = [x for x in line if x]
        # print(earth_tides)
        line = lines[35].split(": ")[1].strip().split("  ")
        self.intrack_thrust = [x for x in line if x]
        # print(intrack_thrust)
        line = lines[36].split(": ")[1].strip().split("   ")
        line = [x for x in line if x]
        self.time_lastob_sta = []
        self.time_lastob_sta.append(datetime.strptime(line[0].replace(" ",""),'%Y/%m/%d%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        self.time_lastob_sta.append(datetime.strptime(line[1].replace(" ",""),'%Y/%m/%d%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        # print(time_lastob_sta)
        line = lines[37].split(": ")[1].strip().split("   ")
        line = [x for x in line if x]
        self.time_lastob_end = []
        self.time_lastob_end.append(datetime.strptime(line[0].replace(" ",""),'%Y/%m/%d%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        self.time_lastob_end.append(datetime.strptime(line[1].replace(" ",""),'%Y/%m/%d%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        # print(time_lastob_end)
        line = lines[38].split(": ")[1].strip().split("  ")
        self.rec_od_span = [float(x) for x in line if x]
        # print(rec_od_span)
        line = lines[39].split(": ")[1].strip().split("  ")
        self.actual_od_span = [float(x) for x in line if x]
        # print(actual_od_span)
        line = lines[40].split(": ")[1].strip().split("  ")
        self.obs_available = [float(x) for x in line if x]
        # print(obs_available)
        line = lines[41].split(": ")[1].strip().split("  ")
        self.obs_used = [float(x) for x in line if x]
        # print(obs_used)
        line = lines[42].split(": ")[1].strip().split("  ")
        self.residuals_accepted = [float(x) for x in line if x]
        # print(residuals_accepted)
        line = lines[43].split(": ")[1].strip().split("  ")
        self.weighted_rms = [float(x) for x in line if x]
        # print(weighted_rms)
        line = lines[44].split(": ")[1].strip().split("  ")
        self.area_pc = [float(x) for x in line if x]
        # print(area_pc)
        line = lines[45].split(": ")[1].strip().split("  ")
        self.area_drag = [float(x) for x in line if x]
        # print(area_drag)
        line = lines[46].split(": ")[1].strip().split("  ")
        self.area_srp = [float(x) for x in line if x]
        # print(area_srp)
        line = lines[47].split(": ")[1].strip().split("  ")
        self.mass = [float(x) for x in line if x]
        # print(mass)
        line = lines[48].split(": ")[1].strip().split("  ")
        self.cd_am = [float(x) for x in line if x]
        # print(cd_am)
        line = lines[49].split(": ")[1].strip().split("  ")
        self.cr_am = [float(x) for x in line if x]
        # print(cr_am)
        line = lines[50].split(": ")[1].strip().split("  ")
        self.thrust_acc = [float(x) for x in line if x]
        # print(thrust_acc)
        line = lines[51].split(": ")[1].strip().split("  ")
        self.sedr = [float(x) for x in line if x]
        # print(sedr)
        line = lines[52].split(": ")[1].strip().split("  ")
        X = [float(x) for x in line if x]
        # print(X)
        line = lines[53].split(": ")[1].strip().split("  ")
        Y = [float(x) for x in line if x]
        # print(Y)
        line = lines[54].split(": ")[1].strip().split("  ")
        Z = [float(x) for x in line if x]
        # print(Z)
        self.position_XYZ = np.array([X,Y,Z])
        line = lines[55].split(": ")[1].strip().split("  ")
        X_dot = [float(x) for x in line if x]
        # print(X_dot)
        line = lines[56].split(": ")[1].strip().split("  ")
        Y_dot = [float(x) for x in line if x]
        # print(Y_dot)
        line = lines[57].split(": ")[1].strip().split("  ")
        Z_dot = [float(x) for x in line if x]
        # print(Z_dot)
        self.position_dot_XYZ = np.array([X_dot,Y_dot,Z_dot])

        line = lines[59].split(": ")[1].strip().split("  ")
        self.apogee = [float(x) for x in line if x]
        # print(apogee)
        line = lines[60].split(": ")[1].strip().split("  ")
        self.perigee = [float(x) for x in line if x]
        # print(perigee)
        line = lines[61].split(": ")[1].strip().split("  ")
        self.eccentricity = [float(x) for x in line if x]
        # print(eccentricity)
        line = lines[62].split(": ")[1].strip().split("  ")
        self.inclination = [float(x) for x in line if x]
        # print(inclination)

        line = lines[64].split(": ")[1].strip().split()
        self.RTN_1sigma = [float(x) for x in line if x]
        # print(RTN_1sigma)

        line = lines[66].split(": ")[1].strip().split()
        RTN_covariance_temp = [float(x) for x in line if x]
        line = lines[67].strip().split()
        RTN_covariance_temp.append([float(x) for x in line if x])
        line = lines[68].strip().split()
        RTN_covariance_temp.append([float(x) for x in line if x])
        self.RTN_covariance = np.zeros((3,3,2))
        self.RTN_covariance[:,:,0] = np.array([ [RTN_covariance_temp[0],RTN_covariance_temp[2][0],RTN_covariance_temp[3][0]],
                                    [RTN_covariance_temp[2][0],RTN_covariance_temp[2][1],RTN_covariance_temp[3][1]],
                                    [RTN_covariance_temp[3][0],RTN_covariance_temp[3][1],RTN_covariance_temp[3][2] ] ])
        self.RTN_covariance[:,:,1] = np.array([ [RTN_covariance_temp[1],RTN_covariance_temp[2][2],RTN_covariance_temp[3][3]],
                                    [RTN_covariance_temp[2][2],RTN_covariance_temp[2][3],RTN_covariance_temp[3][4]],
                                    [RTN_covariance_temp[3][3],RTN_covariance_temp[3][4],RTN_covariance_temp[3][5] ] ] )
    
    def load_cdm_from_mysql(self, cdm_id=str, username=str, password=str):

        db_connection_str = 'mysql+pymysql://'+ username + ':' + password + '@flp1.irs.uni-stuttgart.de:3308/Space_Track'
        db_connection = create_engine(db_connection_str)

        text = 'SELECT * FROM CDM WHERE CDM_ID = ' + cdm_id + ' LIMIT 1'
        mydf = pd.read_sql(text, con=db_connection)

        self.creation_date = datetime.strptime(str(mydf['CREATION_DATE'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc)
        self.originator = mydf['ORIGINATOR'].values
        self.message_for = mydf['MESSAGE_FOR'].values
        self.message_id = mydf['MESSAGE_ID'].values
        
        self.tca = datetime.strptime(str(mydf['TCA'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc)
        self.miss_distance = mydf['MISS_DISTANCE'].values
        self.relative_speed = mydf['RELATIVE_SPEED'].values
        rel_position_R = mydf['RELATIVE_POSITION_R'].values
        rel_position_T = mydf['RELATIVE_POSITION_T'].values
        rel_position_N = mydf['RELATIVE_POSITION_N'].values
        self.rel_position_RTN = np.array([[rel_position_R],[rel_position_T],[rel_position_N]])
        rel_velocity_R = mydf['RELATIVE_VELOCITY_R'].values
        rel_velocity_T = mydf['RELATIVE_VELOCITY_T'].values
        rel_velocity_N = mydf['RELATIVE_VELOCITY_N'].values
        self.rel_velocity_RTN = np.array([[rel_velocity_R],[rel_velocity_T],[rel_velocity_N]])
        self.collision_prob = mydf['COLLISION_PROBABILITY'].values
        self.collision_prob_method = mydf['COLLISION_PROBABILITY_METHOD'].values

        self.object_designator = [mydf['SAT1_OBJECT_DESIGNATOR'].values, mydf['SAT2_OBJECT_DESIGNATOR'].values]
        self.object_name = [mydf['SAT1_OBJECT_NAME'].values, mydf['SAT2_OBJECT_NAME'].values]
        self.itn_designator = [mydf['SAT1_INTERNATIONAL_DESIGNATOR'].values, mydf['SAT2_INTERNATIONAL_DESIGNATOR'].values]
        self.object_type = [mydf['SAT1_OBJECT_TYPE'].values, mydf['SAT2_OBJECT_TYPE'].values]
        self.operator_organization = [mydf['SAT1_OPERATOR_ORGANIZATION'].values, mydf['SAT2_OPERATOR_ORGANIZATION'].values]
        self.ephemeris_name = [mydf['SAT1_EPHEMERIS_NAME'].values, mydf['SAT2_EPHEMERIS_NAME'].values]
        self.maneuverable = [mydf['SAT1_MANEUVERABLE'].values, mydf['SAT2_MANEUVERABLE'].values]
        self.ref_frame = [mydf['SAT1_REF_FRAME'].values, mydf['SAT2_REF_FRAME'].values]
        self.gravity_model = [mydf['SAT1_GRAVITY_MODEL'].values, mydf['SAT2_GRAVITY_MODEL'].values]
        self.atmospheric_model = [mydf['SAT1_ATMOSPHERIC_MODEL'].values, mydf['SAT2_ATMOSPHERIC_MODEL'].values]
        self.n_body_perturbations = [mydf['SAT1_N_BODY_PERTURBATIONS'].values, mydf['SAT2_N_BODY_PERTURBATIONS'].values]
        self.solar_rad_pressure = [mydf['SAT1_SOLAR_RAD_PRESSURE'].values, mydf['SAT2_SOLAR_RAD_PRESSURE'].values]
        self.earth_tides = [mydf['SAT1_EARTH_TIDES'].values, mydf['SAT2_EARTH_TIDES'].values]
        self.intrack_thrust = [mydf['SAT1_INTRACK_THRUST'].values, mydf['SAT2_INTRACK_THRUST']]
        self.time_lastob_sta = []
        self.time_lastob_sta.append(datetime.strptime(str(mydf['SAT1_TIME_LASTOB_START'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        self.time_lastob_sta.append(datetime.strptime(str(mydf['SAT2_TIME_LASTOB_START'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        self.time_lastob_end = []
        self.time_lastob_sta.append(datetime.strptime(str(mydf['SAT1_TIME_LASTOB_END'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        self.time_lastob_sta.append(datetime.strptime(str(mydf['SAT2_TIME_LASTOB_END'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        self.rec_od_span = [mydf['SAT1_RECOMMENDED_OD_SPAN'].values, mydf['SAT2_RECOMMENDED_OD_SPAN'].values]
        self.actual_od_span = [mydf['SAT1_ACTUAL_OD_SPAN'].values, mydf['SAT2_ACTUAL_OD_SPAN'].values]
        self.obs_available = [mydf['SAT1_OBS_AVAILABLE'].values, mydf['SAT2_OBS_AVAILABLE'].values]
        self.obs_used = [mydf['SAT1_OBS_USED'].values, mydf['SAT2_OBS_USED'].values]
        self.residuals_accepted = [mydf['SAT1_RESIDUALS_ACCEPTED'].values, mydf['SAT2_RESIDUALS_ACCEPTED'].values]
        self.weighted_rms = [mydf['SAT1_WEIGHTED_RMS'].values, mydf['SAT2_WEIGHTED_RMS'].values]
        self.area_pc = [mydf['SAT1_AREA_PC'].values[0], mydf['SAT2_AREA_PC'].values[0]]
        self.cd_am = [mydf['SAT1_CD_AREA_OVER_MASS'].values, mydf['SAT2_CD_AREA_OVER_MASS'].values]
        self.cr_am = [mydf['SAT1_CR_AREA_OVER_MASS'].values, mydf['SAT2_CR_AREA_OVER_MASS'].values]
        self.thrust_acc = [mydf['SAT1_THRUST_ACCELERATION'].values, mydf['SAT2_THRUST_ACCELERATION'].values]
        self.sedr = [mydf['SAT1_SEDR'].values, mydf['SAT2_SEDR'].values]
        X = np.array([mydf['SAT1_X'].values, mydf['SAT2_X'].values]).flatten()
        Y = np.array([mydf['SAT1_Y'].values, mydf['SAT2_Y'].values]).flatten()
        Z = np.array([mydf['SAT1_Z'].values, mydf['SAT2_Z'].values]).flatten()
        self.position_XYZ = np.array([X,Y,Z])
        X_dot = np.array([mydf['SAT1_X_DOT'].values, mydf['SAT2_X_DOT'].values]).flatten()
        Y_dot = np.array([mydf['SAT1_Y_DOT'].values, mydf['SAT2_Y_DOT'].values]).flatten()
        Z_dot = np.array([mydf['SAT1_Z_DOT'].values, mydf['SAT2_Z_DOT'].values]).flatten()
        self.position_dot_XYZ = np.array([X_dot,Y_dot,Z_dot])

        self.apogee = [float(mydf['SAT1_COMMENT_APOGEE'].values[0][18:21]), float(mydf['SAT2_COMMENT_APOGEE'].values[0][18:21])]
        self.perigee = [float(mydf['SAT1_COMMENT_PERIGEE'].values[0][19:22]), float(mydf['SAT2_COMMENT_PERIGEE'].values[0][19:22])]
        self.inclination = [float(mydf['SAT1_COMMENT_INCLINATION'].values[0][14:18]), float(mydf['SAT2_COMMENT_INCLINATION'].values[0][14:18])]
        self.eccentricity = [(self.apogee[0]-self.perigee[0])/(self.apogee[0]+self.perigee[0]), (self.apogee[1]-self.perigee[1])/(self.apogee[1]+self.perigee[1])]
        RTN_covariance1 = np.array([[mydf['SAT1_CR_R'].values,mydf['SAT1_CT_R'].values,mydf['SAT1_CN_R'].values],
                                    [mydf['SAT1_CT_R'].values,mydf['SAT1_CT_T'].values,mydf['SAT1_CN_T'].values],
                                    [mydf['SAT1_CN_R'].values,mydf['SAT1_CN_R'].values,mydf['SAT1_CN_N'].values ] ])
        RTN_covariance2 = np.array([[mydf['SAT2_CR_R'].values,mydf['SAT2_CT_R'].values,mydf['SAT2_CN_R'].values],
                                    [mydf['SAT2_CT_R'].values,mydf['SAT2_CT_T'].values,mydf['SAT2_CN_T'].values],
                                    [mydf['SAT2_CN_R'].values,mydf['SAT2_CN_R'].values,mydf['SAT2_CN_N'].values ] ])
        self.RTN_covariance = np.append(RTN_covariance1,RTN_covariance2,axis=2,)

def load_latest_cdm_from_mysql(self):
        db_connection_str = 'mysql+pymysql://turcof:JdD4CJVqvM54yAeXUry4@flp1.irs.uni-stuttgart.de:3308/Space_Track'
        db_connection = create_engine(db_connection_str)

        mydf = pd.read_sql('SELECT * FROM CDM ORDER BY CREATION_DATE DESC LIMIT 1', con=db_connection)

        self.creation_date = datetime.strptime(str(mydf['CREATION_DATE'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc)
        self.originator = mydf['ORIGINATOR'].values
        self.message_for = mydf['MESSAGE_FOR'].values
        self.message_id = mydf['MESSAGE_ID'].values
        
        self.tca = datetime.strptime(str(mydf['TCA'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc)
        self.miss_distance = mydf['MISS_DISTANCE'].values
        self.relative_speed = mydf['RELATIVE_SPEED'].values
        rel_position_R = mydf['RELATIVE_POSITION_R'].values
        rel_position_T = mydf['RELATIVE_POSITION_T'].values
        rel_position_N = mydf['RELATIVE_POSITION_N'].values
        self.rel_position_RTN = np.array([[rel_position_R],[rel_position_T],[rel_position_N]])
        rel_velocity_R = mydf['RELATIVE_VELOCITY_R'].values
        rel_velocity_T = mydf['RELATIVE_VELOCITY_T'].values
        rel_velocity_N = mydf['RELATIVE_VELOCITY_N'].values
        self.rel_velocity_RTN = np.array([[rel_velocity_R],[rel_velocity_T],[rel_velocity_N]])
        self.collision_prob = mydf['COLLISION_PROBABILITY'].values
        self.collision_prob_method = mydf['COLLISION_PROBABILITY_METHOD'].values

        self.object_designator = [mydf['SAT1_OBJECT_DESIGNATOR'].values, mydf['SAT2_OBJECT_DESIGNATOR'].values]
        self.object_name = [mydf['SAT1_OBJECT_NAME'].values, mydf['SAT2_OBJECT_NAME'].values]
        self.itn_designator = [mydf['SAT1_INTERNATIONAL_DESIGNATOR'].values, mydf['SAT2_INTERNATIONAL_DESIGNATOR'].values]
        self.object_type = [mydf['SAT1_OBJECT_TYPE'].values, mydf['SAT2_OBJECT_TYPE'].values]
        self.operator_organization = [mydf['SAT1_OPERATOR_ORGANIZATION'].values, mydf['SAT2_OPERATOR_ORGANIZATION'].values]
        self.ephemeris_name = [mydf['SAT1_EPHEMERIS_NAME'].values, mydf['SAT2_EPHEMERIS_NAME'].values]
        self.maneuverable = [mydf['SAT1_MANEUVERABLE'].values, mydf['SAT2_MANEUVERABLE'].values]
        self.ref_frame = [mydf['SAT1_REF_FRAME'].values, mydf['SAT2_REF_FRAME'].values]
        self.gravity_model = [mydf['SAT1_GRAVITY_MODEL'].values, mydf['SAT2_GRAVITY_MODEL'].values]
        self.atmospheric_model = [mydf['SAT1_ATMOSPHERIC_MODEL'].values, mydf['SAT2_ATMOSPHERIC_MODEL'].values]
        self.n_body_perturbations = [mydf['SAT1_N_BODY_PERTURBATIONS'].values, mydf['SAT2_N_BODY_PERTURBATIONS'].values]
        self.solar_rad_pressure = [mydf['SAT1_SOLAR_RAD_PRESSURE'].values, mydf['SAT2_SOLAR_RAD_PRESSURE'].values]
        self.earth_tides = [mydf['SAT1_EARTH_TIDES'].values, mydf['SAT2_EARTH_TIDES'].values]
        self.intrack_thrust = [mydf['SAT1_INTRACK_THRUST'].values, mydf['SAT2_INTRACK_THRUST']]
        self.time_lastob_sta = []
        self.time_lastob_sta.append(datetime.strptime(str(mydf['SAT1_TIME_LASTOB_START'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        self.time_lastob_sta.append(datetime.strptime(str(mydf['SAT2_TIME_LASTOB_START'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        self.time_lastob_end = []
        self.time_lastob_sta.append(datetime.strptime(str(mydf['SAT1_TIME_LASTOB_END'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        self.time_lastob_sta.append(datetime.strptime(str(mydf['SAT2_TIME_LASTOB_END'].values)[2:-5],'%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc))
        self.rec_od_span = [mydf['SAT1_RECOMMENDED_OD_SPAN'].values, mydf['SAT2_RECOMMENDED_OD_SPAN'].values]
        self.actual_od_span = [mydf['SAT1_ACTUAL_OD_SPAN'].values, mydf['SAT2_ACTUAL_OD_SPAN'].values]
        self.obs_available = [mydf['SAT1_OBS_AVAILABLE'].values, mydf['SAT2_OBS_AVAILABLE'].values]
        self.obs_used = [mydf['SAT1_OBS_USED'].values, mydf['SAT2_OBS_USED'].values]
        self.residuals_accepted = [mydf['SAT1_RESIDUALS_ACCEPTED'].values, mydf['SAT2_RESIDUALS_ACCEPTED'].values]
        self.weighted_rms = [mydf['SAT1_WEIGHTED_RMS'].values, mydf['SAT2_WEIGHTED_RMS'].values]
        self.area_pc = [mydf['SAT1_AREA_PC'].values[0], mydf['SAT2_AREA_PC'].values[0]]
        self.cd_am = [mydf['SAT1_CD_AREA_OVER_MASS'].values, mydf['SAT2_CD_AREA_OVER_MASS'].values]
        self.cr_am = [mydf['SAT1_CR_AREA_OVER_MASS'].values, mydf['SAT2_CR_AREA_OVER_MASS'].values]
        self.thrust_acc = [mydf['SAT1_THRUST_ACCELERATION'].values, mydf['SAT2_THRUST_ACCELERATION'].values]
        self.sedr = [mydf['SAT1_SEDR'].values, mydf['SAT2_SEDR'].values]
        X = np.array([mydf['SAT1_X'].values, mydf['SAT2_X'].values]).flatten()
        Y = np.array([mydf['SAT1_Y'].values, mydf['SAT2_Y'].values]).flatten()
        Z = np.array([mydf['SAT1_Z'].values, mydf['SAT2_Z'].values]).flatten()
        self.position_XYZ = np.array([X,Y,Z])
        X_dot = np.array([mydf['SAT1_X_DOT'].values, mydf['SAT2_X_DOT'].values]).flatten()
        Y_dot = np.array([mydf['SAT1_Y_DOT'].values, mydf['SAT2_Y_DOT'].values]).flatten()
        Z_dot = np.array([mydf['SAT1_Z_DOT'].values, mydf['SAT2_Z_DOT'].values]).flatten()
        self.position_dot_XYZ = np.array([X_dot,Y_dot,Z_dot])

        self.apogee = [float(mydf['SAT1_COMMENT_APOGEE'].values[0][18:21]), float(mydf['SAT2_COMMENT_APOGEE'].values[0][18:21])]
        self.perigee = [float(mydf['SAT1_COMMENT_PERIGEE'].values[0][19:22]), float(mydf['SAT2_COMMENT_PERIGEE'].values[0][19:22])]
        self.inclination = [float(mydf['SAT1_COMMENT_INCLINATION'].values[0][14:18]), float(mydf['SAT2_COMMENT_INCLINATION'].values[0][14:18])]
        self.eccentricity = [(self.apogee[0]-self.perigee[0])/(self.apogee[0]+self.perigee[0]), (self.apogee[1]-self.perigee[1])/(self.apogee[1]+self.perigee[1])]
        RTN_covariance1 = np.array([[mydf['SAT1_CR_R'].values,mydf['SAT1_CT_R'].values,mydf['SAT1_CN_R'].values],
                                    [mydf['SAT1_CT_R'].values,mydf['SAT1_CT_T'].values,mydf['SAT1_CN_T'].values],
                                    [mydf['SAT1_CN_R'].values,mydf['SAT1_CN_R'].values,mydf['SAT1_CN_N'].values ] ])
        RTN_covariance2 = np.array([[mydf['SAT2_CR_R'].values,mydf['SAT2_CT_R'].values,mydf['SAT2_CN_R'].values],
                                    [mydf['SAT2_CT_R'].values,mydf['SAT2_CT_T'].values,mydf['SAT2_CN_T'].values],
                                    [mydf['SAT2_CN_R'].values,mydf['SAT2_CN_R'].values,mydf['SAT2_CN_N'].values ] ])
        self.RTN_covariance = np.append(RTN_covariance1,RTN_covariance2,axis=2,)