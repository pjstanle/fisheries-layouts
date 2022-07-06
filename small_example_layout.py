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
import matplotlib.pyplot as plt

import floris.tools as wfct
import floris.tools.wind_rose as rose

import time
import pyoptsparse
    
def calc_spacing(input_dict):

    turbine_x = input_dict["turbine_x"]
    turbine_y = input_dict["turbine_y"]

    #calculate the spacing between each turbine and every other turbine (without repeating)
    nturbs = len(turbine_x)
    npairs = int((nturbs*(nturbs-1))/2)
    spacing = np.zeros(npairs)

    ind = 0
    for i in range(nturbs):
        for j in range(i,nturbs):
            if i != j:
                spacing[ind] = np.sqrt((turbine_x[i]-turbine_x[j])**2+(turbine_y[i]-turbine_y[j])**2)
                ind += 1

    return spacing


def calc_AEP(input_dict):

    # calculate the wind farm AEP as a function of the grid design variables

    global floris_model
    global function_calls
    global wd
    global ws
    global wf

    turbine_x = input_dict["turbine_x"]
    turbine_y = input_dict["turbine_y"]

    # objective
    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))
    AEP = floris_model.get_farm_AEP(wd,ws,wf)

    return AEP


def obj_func(input_dict):

    # calculate the wind farm AEP as a function of the grid design variables

    global floris_model
    global function_calls
    global wd
    global ws
    global wf
    global min_spacing
    global scale

    # print("objective function")
    funcs = {}

    # objective
    AEP = calc_AEP(input_dict)
    funcs["aep_obj"] = -AEP/scale

    # spcaing constraint
    D = floris_model.floris.farm.turbines[0].rotor_diameter
    funcs["spacing_con"] = calc_spacing(input_dict) - D*min_spacing
    
    fail = False
    return funcs, fail


if __name__=="__main__":

    global floris_model
    global wd
    global ws
    global wf
    global min_spacing
    global scale

    wd = np.linspace(0.0,350.0,36)
    ws = np.ones(len(wd))*10.5
    wf = np.array([0.02557391, 0.02370852, 0.02518278, 0.02334747, 0.02379878,
       0.02196347, 0.02057947, 0.01811234, 0.01447182, 0.01612661,
       0.01287722, 0.01209495, 0.01573548, 0.01714956, 0.01805217,
       0.01973704, 0.01931582, 0.01994765, 0.02903391, 0.0308993 ,
       0.03974486, 0.04729669, 0.0500346 , 0.06020399, 0.06080573,
       0.05274243, 0.03866173, 0.03610434, 0.02668713, 0.02557391,
       0.025604  , 0.02632608, 0.02819147, 0.02746939, 0.02409965,
       0.02274574])

    scale = 1E8

    floris_model = wfct.floris_interface.FlorisInterface("iea_model_25.json")
    floris_model.set_gch(False)

    # wd = np.array([270.0])
    # ws = np.array([10.0])
    # wf = np.array([1.0])

    nturbs = 4 # if you want to limit the number of turbines in the farm

    avg_spacing = 7.0 # avgerage spacing in rotor diameters
    D = floris_model.floris.farm.turbines[0].rotor_diameter
    side = (np.sqrt(nturbs)-1.0)*D*avg_spacing
    print("side length: ", side)

    min_spacing = 4.0 # minimum turbine spacing in rotor diameters

    xmin = 0.0
    xmax = side
    ymin = 0.0
    ymax = side

    nruns = 10
    best_AEP = 0.0
    plot_results = True
    
    for i in range(nruns):

        start_x = np.random.rand(nturbs)*side
        start_y = np.random.rand(nturbs)*side

        start = {}
        start["turbine_x"] = start_x
        start["turbine_y"] = start_y
        start_AEP = calc_AEP(start)/scale
        print("start AEP %s: "%i, start_AEP)

        start_time = time.time()
        optProb = pyoptsparse.Optimization("LayoutOpt",obj_func)
        optProb.addVarGroup("turbine_x",nturbs,type="c",lower=xmin,upper=xmax,value=start_x)
        optProb.addVarGroup("turbine_y",nturbs,type="c",lower=ymin,upper=ymax,value=start_y)
        npairs = int((nturbs*(nturbs-1))/2)
        optProb.addConGroup("spacing_con",npairs,lower=0.0)

        optProb.addObj("aep_obj")
        optimize = pyoptsparse.SNOPT()
        print("start optimization %s"%i)
        solution = optimize(optProb)
        
        opt = solution.getDVs()
        opt_x = solution.getDVs()["turbine_x"]
        opt_y = solution.getDVs()["turbine_y"]
        opt_AEP = calc_AEP(opt)
        opt_spacing = calc_spacing(opt)
        print("optimized AEP %s: "%i, opt_AEP/scale)

        eps = 1E-3

        if opt_AEP > best_AEP and np.max(opt_x)<(xmax+eps) and np.max(opt_y)<(ymax+eps) and np.min(opt_x)>(xmin-eps) and np.min(opt_y)>(ymin-eps) and np.min(opt_spacing)>(min_spacing*D-eps):
            best_AEP = opt_AEP
            best_x = opt_x
            best_y = opt_y
            
        print("current best_AEP: ", best_AEP)
        print("current best_x: ", best_x)
        print("current best_y: ", best_y)
        


    if plot_results == True:
        plt.figure(1,figsize=(6,6))

        bx = np.array([xmin,xmax,xmax,xmin,xmin])
        by = np.array([ymin,ymin,ymax,ymax,ymin])
        plt.plot(bx,by,"--k",linewidth=0.5)

        floris_model.reinitialize_flow_field(layout_array=(best_x,best_y))
        index = np.argmax(wf)
        floris_model.reinitialize_flow_field(wind_direction=wd[index])
        floris_model.reinitialize_flow_field(wind_speed=ws[index])
        floris_model.calculate_wake()
        hor_plane = floris_model.get_hor_plane()

        # Plot and show
        # fig, ax = plt.subplots()
        ax = plt.gca()
        wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
        wfct.visualization.plot_turbines(ax,best_x,best_y,np.zeros(len(best_x)),D,wind_direction=wd[index])

        plt.figure(2,figsize=(4,4))
        plt.plot(bx,by,"--k",linewidth=0.5)

        for i in range(nturbs):
            circ = plt.Circle((best_x[i],best_y[i]),radius=D/2.0,color="C1")
            plt.gca().add_patch(circ)
        
        plt.axis("equal")

        plt.show()


                    
