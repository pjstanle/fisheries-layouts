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
from shapely import geometry


def create_initial_grid(x_spacing,y_spacing,nturbs=53):
    global nrows

    ncols = int(np.floor(nturbs/nrows))
    extra_turbs = nturbs%(nrows*ncols)

    xlocs = np.arange(ncols)*x_spacing
    ylocs = np.arange(nrows)*y_spacing

    # make initial grid
    layout_x = np.array([x for x in xlocs for y in ylocs])
    layout_y = np.array([y for x in xlocs for y in ylocs])

    # add on extra turbines
    layout_x = np.append(layout_x,np.zeros(extra_turbs)+x_spacing*ncols)
    layout_y = np.append(layout_y,(nrows-np.linspace(extra_turbs,1,extra_turbs))*y_spacing)

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


def get_xy(A):
    x = np.zeros(len(A))
    y = np.zeros(len(A))
    for i in range(len(A)):
        x[i] = A[i][0]
        y[i] = A[i][1]
    return x,y


def plot_poly(geom,ax):
    if geom.type == 'Polygon':
        exterior_coords = geom.exterior.coords[:]
        x,y = get_xy(exterior_coords)
        ax.plot(x,y,"--k",linewidth=0.5)

        for interior in geom.interiors:
            interior_coords = interior.coords[:]
            x,y = get_xy(interior_coords)
            ax.plot(x,y,"b")

    elif geom.type == 'MultiPolygon':

        for part in geom:
            exterior_coords = part.exterior.coords[:]
            x,y = get_xy(exterior_coords)
            ax.plot(x,y,"k")
            for interior in part.interiors:
                interior_coords = interior.coords[:]
                x,y = get_xy(interior_coords)
                ax.plot(x,y,"b")


def evaluate_plant(x):
    global floris_model
    global wd
    global ws
    global wf
    global base_x
    global base_y
    global nrows
    global rotor_diameter
    global poly
    global poly_line
    global center_x
    global center_y

    x_spacing = x["x_spacing"]
    y_spacing = x["y_spacing"]
    rotation = x["rotation"]
    shear = x["shear"]

    base_x,base_y = create_initial_grid(x_spacing,y_spacing)
    shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)
    middle_x = (np.max(turbine_x)+np.min(turbine_x))/2.0 
    middle_y = (np.max(turbine_y)+np.min(turbine_y))/2.0 
    dx = center_x-middle_x
    dy = center_y-middle_y
    turbine_x = turbine_x + dx
    turbine_y = turbine_y + dy

    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))

    plt.cla()
    plot_poly(poly,plt.gca())
    plot_turbines(turbine_x,turbine_y,rotor_diameter/2.0,"C0")
    plt.axis("equal")
    plt.pause(0.01)


    AEP = floris_model.get_farm_AEP_parallel(wd, ws, wf)

    constraint_array = np.zeros(5)
    mnx = np.argmin(turbine_x)
    mny = np.argmin(turbine_y)
    mxx = np.argmax(turbine_x)
    mxy = np.argmax(turbine_y)
    pt1 = geometry.Point(turbine_x[mnx],turbine_y[mnx])
    pt2 = geometry.Point(turbine_x[mxx],turbine_y[mxx])
    pt3 = geometry.Point(turbine_x[mny],turbine_y[mny])
    pt4 = geometry.Point(turbine_x[mxy],turbine_y[mxy])
    pt5 = geometry.Point(turbine_x[-2],turbine_y[-2])

    d = pt1.distance(poly_line)
    if pt1.within(poly):
        d = d*-1
    constraint_array[0] = d

    d = pt2.distance(poly_line)
    if pt2.within(poly):
        d = d*-1
    constraint_array[1] = d

    d = pt3.distance(poly_line)
    if pt3.within(poly):
        d = d*-1
    constraint_array[2] = d

    d = pt4.distance(poly_line)
    if pt4.within(poly):
        d = d*-1
    constraint_array[3] = d

    d = pt5.distance(poly_line)
    if pt5.within(poly):
        d = d*-1
    constraint_array[4] = d

    funcs = {}
    fail = False
    funcs["obj"] = -AEP/1E12
    funcs["constraint"] = np.max(constraint_array)
    
    return funcs, fail


if __name__=="__main__":

    global nrows
    global rotor_diameter
    global poly
    global poly_line
    global center_x
    global center_y

    nrows = 6

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

    pt1 = geometry.Point(-2770.858882647537, 8622.988978149027)
    pt2 = geometry.Point(-158.92378962857697, -68.42766176234385)
    pt3 = geometry.Point(12262.352214377384, 5279.789084423289)
    pt4 = geometry.Point(9650.417121358425, 13971.20572433466)
    poly = geometry.Polygon((pt1,pt2,pt3,pt4))
    poly_line = geometry.LineString((pt1,pt2,pt3,pt4))

    center_x = (12262.352214377384-2770.858882647537)/2
    center_y = (13971.20572433466-68.42766176234385)/2

    print("end setup: ", time.time()-start_setup)

    
    start_spacing = 1640.0062233309463
    start_rotation = np.deg2rad(24.201843023746154)
    start_shear = np.deg2rad(6.406589605537475)

    # OPTIMIZATION
    print("start optimization")
    start_opt = time.time()
    optProb = pyoptsparse.Optimization("optimize 4",evaluate_plant)
    optProb.addVar("spacing",type="c",lower=3*rotor_diameter,upper=None,value=start_spacing)
    optProb.addVar("rotation",type="c",lower=None,upper=None,value=start_rotation)
    optProb.addVar("shear",type="c",lower=-np.pi/2,upper=np.pi/2,value=start_shear)
    optProb.addCon("constraint",lower=None,upper=0.0)

    optProb.addObj("obj")
    optimize = pyoptsparse.SLSQP()
    optimize.setOption("MAXIT",value=50)

    solution = optimize(optProb,sens="FD")
    print("end optimization: ", time.time()-start_opt)

    # END RESULTS
    opt_DVs = solution.getDVs()
    opt_spacing = opt_DVs["spacing"]
    opt_rotation = opt_DVs["rotation"]
    opt_shear = opt_DVs["shear"]
    funcs,fail = evaluate_plant(opt_DVs)
    opt_aep = -funcs["obj"]

    print("optimized AEP: ", opt_aep)
    print("optimal spacing: ", opt_spacing)
    print("optimal rotation: ", np.rad2deg(opt_rotation))
    print("optimal shear: ", np.rad2deg(opt_shear))

    base_x,base_y = create_initial_grid(opt_spacing)
    shear_x,shear_y = shear_grid_locs(opt_shear,base_x,base_y)
    opt_x,opt_y = rotate_grid_locs(opt_rotation,shear_x,shear_y)
    middle_x = (np.max(opt_x)+np.min(opt_x))/2.0 
    middle_y = (np.max(opt_y)+np.min(opt_y))/2.0 
    dx = center_x-middle_x
    dy = center_y-middle_y
    opt_x = opt_x + dx
    opt_y = opt_y + dy
    
    # print("scenario1_x = np."+repr(opt_x))
    # print("scenario1_y = np."+repr(opt_y))
    plot_turbines(opt_x,opt_y,rotor_diameter/2.0,"C0")
    plt.axis("equal")
    plt.show()
    