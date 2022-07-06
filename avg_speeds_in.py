# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation

import numpy as np
import pandas as pd

import floris.tools as wfct
import floris.tools.wind_rose as rose

from shapely.geometry import Polygon
import matplotlib.pyplot as plt

floris_model = wfct.floris_interface.FlorisInterface("12MW.json")
floris_model.set_gch(False)

max_turbs = 170
min_spacing = 2.0

wind_rose = rose.WindRose()
wind_rose.load("138m_data.p")

wd = wind_rose.df.wd
ws = wind_rose.df.ws
wf = wind_rose.df.freq_val

plt.plot(wd,wf,"o")
plt.show()


base_x = np.array([569768.0, 571424.0, 573080.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 584672.0, 586328.0, 587984.0, 
                    589640.0, 569768.0, 571424.0, 573080.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 584672.0, 586328.0, 
                    587984.0, 589640.0, 569768.0, 571424.0, 573080.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 584672.0, 
                    586328.0, 587984.0, 589640.0, 569768.0, 571424.0, 573080.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 
                    584672.0, 586328.0, 587984.0, 589640.0, 569768.0, 571424.0, 573080.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 
                    583016.0, 584672.0, 586328.0, 587984.0, 589640.0, 564800.0, 566456.0, 568112.0, 569768.0, 571424.0, 573080.0, 574736.0, 
                    576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 584672.0, 586328.0, 587984.0, 589640.0, 564800.0, 566456.0, 568112.0, 
                    569768.0, 571424.0, 573080.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 584672.0, 586328.0, 587984.0, 
                    589640.0, 568112.0, 569768.0, 571424.0, 573080.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 584672.0, 
                    586328.0, 587984.0, 589640.0, 569768.0, 571424.0, 573080.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 
                    584672.0, 586328.0, 587984.0, 569768.0, 571424.0, 573080.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 
                    584672.0, 586328.0, 569768.0, 571424.0, 573080.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 584672.0, 
                    586328.0, 571424.0, 573080.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 584672.0, 573080.0, 574736.0, 
                    576392.0, 578048.0, 579704.0, 581360.0, 583016.0, 574736.0, 576392.0, 578048.0, 579704.0, 581360.0, 576392.0, 578048.0, 
                    579704.0, 578048.0])
base_y = np.array([4358400.0, 4358400.0, 4358400.0, 4358400.0, 4358400.0, 4358400.0, 4358400.0, 4358400.0, 4358400.0, 4358400.0, 4358400.0, 
                    4358400.0, 4358400.0, 4356744.0, 4356744.0, 4356744.0, 4356744.0, 4356744.0, 4356744.0, 4356744.0, 4356744.0, 4356744.0, 
                    4356744.0, 4356744.0, 4356744.0, 4356744.0, 4355088.0, 4355088.0, 4355088.0, 4355088.0, 4355088.0, 4355088.0, 4355088.0, 
                    4355088.0, 4355088.0, 4355088.0, 4355088.0, 4355088.0, 4355088.0, 4353432.0, 4353432.0, 4353432.0, 4353432.0, 4353432.0, 
                    4353432.0, 4353432.0, 4353432.0, 4353432.0, 4353432.0, 4353432.0, 4353432.0, 4353432.0, 4351776.0, 4351776.0, 4351776.0, 
                    4351776.0, 4351776.0, 4351776.0, 4351776.0, 4351776.0, 4351776.0, 4351776.0, 4351776.0, 4351776.0, 4351776.0, 4350120.0, 
                    4350120.0, 4350120.0, 4350120.0, 4350120.0, 4350120.0, 4350120.0, 4350120.0, 4350120.0, 4350120.0, 4350120.0, 4350120.0, 
                    4350120.0, 4350120.0, 4350120.0, 4350120.0, 4348464.0, 4348464.0, 4348464.0, 4348464.0, 4348464.0, 4348464.0, 4348464.0, 
                    4348464.0, 4348464.0, 4348464.0, 4348464.0, 4348464.0, 4348464.0, 4348464.0, 4348464.0, 4348464.0, 4346808.0, 4346808.0, 
                    4346808.0, 4346808.0, 4346808.0, 4346808.0, 4346808.0, 4346808.0, 4346808.0, 4346808.0, 4346808.0, 4346808.0, 4346808.0, 
                    4346808.0, 4345152.0, 4345152.0, 4345152.0, 4345152.0, 4345152.0, 4345152.0, 4345152.0, 4345152.0, 4345152.0, 4345152.0, 
                    4345152.0, 4345152.0, 4343496.0, 4343496.0, 4343496.0, 4343496.0, 4343496.0, 4343496.0, 4343496.0, 4343496.0, 4343496.0, 
                    4343496.0, 4343496.0, 4341840.0, 4341840.0, 4341840.0, 4341840.0, 4341840.0, 4341840.0, 4341840.0, 4341840.0, 4341840.0, 
                    4341840.0, 4341840.0, 4340184.0, 4340184.0, 4340184.0, 4340184.0, 4340184.0, 4340184.0, 4340184.0, 4340184.0, 4340184.0, 
                    4338528.0, 4338528.0, 4338528.0, 4338528.0, 4338528.0, 4338528.0, 4338528.0, 4336872.0, 4336872.0, 4336872.0, 4336872.0, 
                    4336872.0, 4335216.0, 4335216.0, 4335216.0, 4333560.0])


# floris_model.reinitialize_flow_field(layout_array=(base_x,base_y))

# import time
# start_par = time.time()
# base_AEP = floris_model.get_farm_AEP_parallel(wd, ws, wf)
# print("parallel: ", time.time()-start_par)


# print("base AEP: ", base_AEP/1E12)
# print("base nturbs: ", len(base_x))
# base_AEP = 9.583697039410454



sub_boundaries = [
    [579200,	4333200],
    [578000,	4333200],
    [576800,	4333200],
    [576800,	4334400],
    [575600,	4334400],
    [575600,	4335600],
    [574400,	4335600],
    [574400,	4336800],
    [573200,	4336800],
    [573200,	4338000],
    [572000,	4338000],
    [572000,	4339200],
    [570800,	4339200],
    [570800,	4340400],
    [569600,	4340400],
    [569600,	4341600],
    [569600,	4342800],
    [569600,	4344000],
    [569600,	4345200],
    [568400,	4345200],
    [568400,	4346400],
    [567200,	4346400],
    [567200,	4347600],
    [566000,	4347600],
    [564800,	4347600],
    [564800,	4348800],
    [564800,	4350000],
    [564800,	4351200],
    [566000,	4351200],
    [567200,	4351200],
    [568400,	4351200],
    [569600,	4351200],
    [569600,	4352400],
    [569600,	4353600],
    [569600,	4354800],
    [569600,	4356000],
    [569600,	4357200],
    [569600,	4358400],
    [570800,	4358400],
    [572000,	4358400],
    [576800,	4358400],#here y_max
    [591200,	4358400],#here 2nd point
    [591200,	4353600],
    [591200,	4348800],
    [591200,	4347600],
    [591200,	4346400],
    [590000,	4346400],
    [590000,	4345200],
    [588800,	4345200],
    [588800,	4344000],
    [587600,	4344000],
    [587600,	4342800],
    [586400,	4342800],
    [586400,	4341600],
    [585200,	4341600],
    [585200,	4340400],
    [585200,	4339200],
    [584000,	4339200],
    [584000,	4338000],
    [582800,	4338000],
    [582800,	4336800],
    [581600,	4336800],
    [581600,	4335600],
    [580400,	4335600],
    [580400,	4334400],
    [579200,	4334400]
]         
poly = Polygon(sub_boundaries)