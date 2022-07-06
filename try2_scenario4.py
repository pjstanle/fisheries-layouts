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

import warnings
warnings.filterwarnings("ignore")


def create_initial_grid(x_spacing,y_spacing,nturbs=53):
    global nrows

    ncols = int(np.floor(nturbs/nrows))
    extra_turbs = nturbs%(nrows*ncols)

    if extra_turbs != 0:
        ncols += 1

    xlocs = np.arange(ncols)*x_spacing
    ylocs = np.arange(nrows)*y_spacing

    # make initial grid
    layout_x = np.array([x for x in xlocs for y in ylocs])
    layout_y = np.array([y for x in xlocs for y in ylocs])

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


def plot_turbines(x,y,r,color="C0",nums=False):
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
    global nturbs

    x_spacing = x["x_spacing"]*rotor_diameter
    y_spacing = x["y_spacing"]*rotor_diameter
    rotation = x["rotation"]
    shear = x["shear"]

    base_x,base_y = create_initial_grid(x_spacing,y_spacing,nturbs=nturbs)
    shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)
    middle_x = (np.max(turbine_x)+np.min(turbine_x))/2.0 
    middle_y = (np.max(turbine_y)+np.min(turbine_y))/2.0 
    dx = center_x-middle_x
    dy = center_y-middle_y
    turbine_x = turbine_x + dx
    turbine_y = turbine_y + dy

    # remove extra turbines
    ncols = len(turbine_x)/nrows
    additional = len(turbine_x)-nturbs
    index = np.arange(additional)+int(nrows*(ncols-1))
    turbine_x = np.delete(turbine_x,index)
    turbine_y = np.delete(turbine_y,index)

    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))

    plt.cla()
    plot_poly(poly,plt.gca())
    plot_turbines(turbine_x,turbine_y,rotor_diameter/2.0)
    plt.gca().autoscale_view()
    plt.axis("equal")
    plt.pause(0.01)


    AEP = floris_model.get_farm_AEP_parallel(wd, ws, wf)

    constraint_array = np.zeros(nturbs)

    for k in range(nturbs):
        pt = geometry.Point(turbine_x[k],turbine_y[k])
        d = pt.distance(poly_line)
        if pt.within(poly):
            d = d*-1
        constraint_array[k] = d


    funcs = {}
    fail = False
    funcs["obj"] = -AEP/1E12

    ncols = int(np.floor(nturbs/nrows))
    extra_turbs = nturbs%(nrows*ncols)
    if extra_turbs != 0:
        ncols += 1
    total_area = x["x_spacing"]*x["y_spacing"]*(nrows-1)*(ncols-1)
    funcs["total_area"] = total_area

    funcs["spacing1"] = np.sqrt((turbine_x[1]-turbine_x[0])**2+(turbine_y[1]-turbine_y[0])**2)
    funcs["spacing2"] = np.sqrt((turbine_x[nrows]-turbine_x[0])**2+(turbine_y[nrows]-turbine_y[0])**2)
    funcs["spacing3"] = np.sqrt((turbine_x[nrows+1]-turbine_x[0])**2+(turbine_y[nrows+1]-turbine_y[0])**2)
    funcs["boundary_constraint"] = np.max(constraint_array)

    return funcs, fail


if __name__=="__main__":

    global nrows
    global rotor_diameter
    global poly
    global poly_line
    global center_x
    global center_y

    nrows = 8
    nturbs = 53

    # INITIAL SETUP
    start_setup = time.time()
    print("start setup")
    floris_model = wfct.floris_interface.FlorisInterface("15MW.json")
    floris_model.set_gch(False)
    rotor_diameter = floris_model.floris.farm.turbines[0].rotor_diameter

    wind_rose = rose.WindRose()
    wind_rose.load("150m_data.p")
    wind_rose.df = wind_rose.resample_wind_direction(wind_rose.df)
    wind_rose.df = wind_rose.resample_average_ws_by_wd(wind_rose.df)

    wd = wind_rose.df.wd
    ws = wind_rose.df.ws
    wf = wind_rose.df.freq_val

    x_spacing = 7.009432488886936
    y_spacing = 6.88944554493901-0.25
    shear = np.deg2rad(-59.95846465731358)
    rotation = np.deg2rad(1.8810675752002097)

    center_x=0
    center_y=0

    min_spacing = 3*rotor_diameter

    pt1 = geometry.Point(6478.376698485294, -6120.629680514268)
    pt2 = geometry.Point(17188.12742169846, -6828.612310049923)
    pt3 = geometry.Point(-6478.3766984852955, 6120.629680514268)
    pt4 = geometry.Point(-17188.12742169846, 6828.612310049923) 
    poly = geometry.Polygon((pt1,pt2,pt3,pt4))
    poly_line = geometry.LineString((pt1,pt2,pt3,pt4))

    # OPTIMIZATION
    print("start optimization")
    start_opt = time.time()
    optProb = pyoptsparse.Optimization("optimize 4",evaluate_plant)
    optProb.addVar("x_spacing",type="c",lower=3,upper=10.0,value=x_spacing)
    optProb.addVar("y_spacing",type="c",lower=3,upper=10.0,value=y_spacing)
    optProb.addVar("rotation",type="c",lower=None,upper=None,value=rotation)
    optProb.addVar("shear",type="c",lower=-3*np.pi/7,upper=3*np.pi/7,value=shear)
    optProb.addCon("total_area",lower=None,upper=2646.0)
    optProb.addCon("spacing1",lower=min_spacing,upper=None)
    optProb.addCon("spacing2",lower=min_spacing,upper=None)
    optProb.addCon("spacing3",lower=min_spacing,upper=None)
    optProb.addCon("boundary_constraint",lower=None,upper=0.0)

    optProb.addObj("obj")
    optimize = pyoptsparse.SLSQP()
    optimize.setOption("MAXIT",value=50)

    solution = optimize(optProb,sens="FD")
    print("end optimization: ", time.time()-start_opt)

    # END RESULTS
    opt_DVs = solution.getDVs()
    opt_xspacing = opt_DVs["x_spacing"]
    opt_yspacing = opt_DVs["y_spacing"]
    opt_rotation = opt_DVs["rotation"]
    opt_shear = opt_DVs["shear"]
    funcs,fail = evaluate_plant(opt_DVs)
    opt_aep = -funcs["obj"]

    print("optimized AEP: ", opt_aep)
    print("optimal x spacing: ", opt_xspacing)
    print("optimal y spacing: ", opt_yspacing)
    print("optimal rotation: ", np.rad2deg(opt_rotation))
    print("optimal shear: ", np.rad2deg(opt_shear))

    print("boundary constraint: ", funcs["boundary_constraint"])

    # opt_xspacing = x_spacing*rotor_diameter
    # opt_yspacing = y_spacing*rotor_diameter
    # opt_rotation = rotation
    # opt_shear = shear

    # x = {}
    # x["x_spacing"] = x_spacing
    # x["y_spacing"] = y_spacing-0.25
    # x["rotation"] = opt_rotation
    # x["shear"] = opt_shear
    # funcs,_ = evaluate_plant(x)
    # print(funcs["boundary_constraint"])
    # print(funcs["obj"])


    base_x,base_y = create_initial_grid(opt_xspacing,opt_yspacing,nturbs=nturbs)
    shear_x,shear_y = shear_grid_locs(opt_shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(opt_rotation,shear_x,shear_y)
    middle_x = (np.max(turbine_x)+np.min(turbine_x))/2.0 
    middle_y = (np.max(turbine_y)+np.min(turbine_y))/2.0 
    dx = center_x-middle_x
    dy = center_y-middle_y
    turbine_x = turbine_x + dx
    turbine_y = turbine_y + dy
    ncols = len(turbine_x)/nrows
    additional = len(turbine_x)-nturbs
    index = np.arange(additional)+int(nrows*(ncols-1))
    turbine_x = np.delete(turbine_x,index)
    turbine_y = np.delete(turbine_y,index)

    plot_poly(poly,plt.gca())
    plot_turbines(turbine_x,turbine_y,rotor_diameter,nums=True)

    plt.axis("equal")
    plt.show()
    