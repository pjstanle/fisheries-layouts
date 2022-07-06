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
import floris.tools as wfct
import floris.tools.wind_rose as rose
import matplotlib.pyplot as plt
from pandas.core import base
import pyoptsparse
import time


def create_initial_grid():
    nrows = 7
    ncols = 9
    x_spacing = 7*rotor_diameter
    y_spacing = 7*rotor_diameter
    xlocs = np.arange(0,ncols)*x_spacing
    ylocs = np.arange(0,nrows)*y_spacing
    row_number = np.arange(0,nrows)

    # make initial grid
    layout_x = np.array([x for x in xlocs for y in ylocs])
    layout_y = np.array([y for x in xlocs for y in ylocs])

    # add on extra turbines
    layout_x = np.append(layout_x,np.zeros(3)+x_spacing*ncols)
    layout_y = np.append(layout_y,(nrows-np.array([3,2,1]))*y_spacing)

    return layout_x, layout_y


def shear_grid_locs(shear,x,y):
    shear_y = np.copy(y)
    shear_x = np.zeros_like(shear_y)
    dy = y[1]-y[0]
    nturbs = len(x)
    for i in range(nturbs):
        row_num = (y[i]-np.min(y))/dy
        shear_x[i] = x[i] + (row_num-1)*dy*np.tan(shear)
    return shear_x, shear_y


def rotate_grid_locs(rotation,x,y):
    # rotate
    rotate_x = np.cos(rotation)*x - np.sin(rotation)*y
    rotate_y = np.sin(rotation)*x + np.cos(rotation)*y

    return rotate_x, rotate_y


def plot_turbines(x,y,r,color,nums=False):
    n = len(x)
    for i in range(n):
        t = plt.Circle((x[i],y[i]),r,color=color)
        plt.gca().add_patch(t)
        if nums==True:
            plt.text(x[i],y[i],"%s"%(i+1))


def evaluate_plant(x):
    global floris_model
    global wd
    global ws
    global wf
    global base_x
    global base_y

    rotation = x["rotation"]
    shear = x["shear"]

    shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)
    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))

    AEP = floris_model.get_farm_AEP_parallel(wd, ws, wf)

    funcs = {}
    fail = False
    funcs["obj"] = -AEP/1E12
    
    return funcs, fail


if __name__=="__main__":

    global base_x
    global base_y

    # INITIAL SETUP
    start_setup = time.time()
    print("start setup")
    floris_model = wfct.floris_interface.FlorisInterface("12MW.json")
    floris_model.set_gch(False)
    rotor_diameter = floris_model.floris.farm.turbines[0].rotor_diameter

    wind_rose = rose.WindRose()
    wind_rose.load("138m_data.p")
    wind_rose.df = wind_rose.resample_wind_direction(wind_rose.df)
    wind_rose.df = wind_rose.resample_average_ws_by_wd(wind_rose.df)

    wd = wind_rose.df.wd
    ws = wind_rose.df.ws
    wf = wind_rose.df.freq_val

    # wind_rose.plot_wind_rose(wd_bins=np.arange(0, 360, 5.0))
    # plt.show()

    base_x,base_y = create_initial_grid()

    plot_turbines(base_x,base_y,rotor_diameter/2,"C0",nums=True)
    plt.axis("equal")
    plt.show()

    # print("end setup: ", time.time()-start_setup)

    # # INITIAL SWEEP
    # run_sweep = False
    # if run_sweep:
    #     print("start sweep")
    #     start_sweep = time.time()
    #     n = 100
    #     a = np.linspace(0,2*np.pi,n)
    #     aep_sweep = np.zeros(n)
    #     x = {}
    #     for i in range(n):
    #         x["rotation"] = a[i]
    #         funcs,_ = evaluate_plant(x)
    #         aep_sweep[i] = -funcs["obj"]
    
    #     # plt.plot(a,aep_sweep)
    #     # plt.show()
    #     start_rotation = a[np.argmax(aep_sweep)]
    #     print("end sweep: ", time.time()-start_sweep)
    # else:
    #     start_rotation = np.pi/4

    # start_shear = 0.0
    # x = {}
    # x["rotation"] = start_rotation
    # x["shear"] = start_shear
    # funcs,_ = evaluate_plant(x)
    # start_aep = -funcs["obj"]
    
    # # OPTIMIZATION
    # print("start optimization")
    # start_opt = time.time()
    # optProb = pyoptsparse.Optimization("optimize rotation",evaluate_plant)
    # optProb.addVar("rotation",type="c",lower=-100,upper=100,value=start_rotation)
    # optProb.addVar("shear",type="c",lower=-np.pi/2,upper=np.pi/2,value=start_shear)

    # optProb.addObj("obj")
    # optimize = pyoptsparse.SLSQP()
    # optimize.setOption("MAXIT",value=50)

    # solution = optimize(optProb,sens="FD")
    # print("end optimization: ", time.time()-start_opt)

    # # END RESULTS
    # opt_DVs = solution.getDVs()
    # opt_rotation = opt_DVs["rotation"]
    # opt_shear = opt_DVs["shear"]
    # funcs,fail = evaluate_plant(opt_DVs)
    # opt_aep = -funcs["obj"]

    # print("start AEP: ", start_aep)
    # print("optimized AEP: ", opt_aep)
    # print("percent improvement: ", (opt_aep-start_aep)/start_aep*100.0)
    # print("optimal rotation: ", np.rad2deg(opt_rotation))
    # print("optimal shear: ", np.rad2deg(opt_shear))

    # shear_x,shear_y = shear_grid_locs(opt_shear,base_x,base_y)
    # opt_x,opt_y = rotate_grid_locs(opt_rotation,shear_x,shear_y)
    
    # print("scenario1_x = np."+repr(opt_x))
    # print("scenario1_y = np."+repr(opt_y))
    # plot_turbines(opt_x,opt_y,rotor_diameter/2.0,"C0")
    # plt.axis("equal")
    # plt.show()
    