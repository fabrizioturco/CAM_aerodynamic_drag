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
