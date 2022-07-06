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
import time
from shapely import geometry
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def calc_spacing(turbine_x,turbine_y):

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


def create_initial_grid_scenario1(nrows,nturbs,x_spacing,y_spacing):

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


def evaluate_plant_scenario1(x):
    global floris_model
    global wd
    global ws
    global wf
    global base_x
    global base_y
    global nrows
    global nturbs
    global center_x
    global center_y

    x_spacing = x["x_spacing"]
    y_spacing = x["y_spacing"]
    rotation = x["rotation"]
    shear = x["shear"]

    base_x,base_y = create_initial_grid_scenario1(nrows,nturbs,x_spacing,y_spacing)
    shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)

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


    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))

    AEP = floris_model.get_farm_AEP_parallel(wd, ws, wf)

    funcs = {}
    fail = False
    funcs["obj"] = -AEP/1E12
    
    return funcs, fail


def create_initial_grid_scenario2(nrows,nturbs,x_spacing,y_spacing):

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


def evaluate_plant_scenario2(x):
    global floris_model
    global wd
    global ws
    global wf
    global poly
    global poly_line
    global center_x
    global center_y
    global nrows
    global nturbs

    rotation = x["rotation"]
    shear = x["shear"]
    x_spacing = x["x_spacing"]
    y_spacing = x["y_spacing"]

    base_x, base_y = create_initial_grid_scenario2(nrows,nturbs,x_spacing,y_spacing)
    shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)

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

    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))

    AEP = floris_model.get_farm_AEP_parallel(wd, ws, wf)

    funcs = {}
    fail = False
    funcs["obj"] = -AEP/1E12
    
    return funcs, fail


def create_initial_grid_scenario2_shipping(nrows,nturbs,x_spacing,y_spacing):

    ncols = int(np.floor(nturbs/nrows))
    extra_turbs = nturbs%(nrows*ncols)

    if extra_turbs != 0:
        ncols += 1

    ncols_right = int(ncols/2)
    if ncols%2 == 0:
        ncols_left = int(ncols_right)
    else: 
        ncols_left = int(ncols_right+1)

    nm = 3704/2
    xlocs_left = np.arange(ncols_left)*x_spacing
    ylocs_left = np.arange(nrows)*y_spacing

    shipping_lane_width = 2*nm
    xlocs_right = np.arange(ncols_right)*x_spacing + max(xlocs_left) + shipping_lane_width
    ylocs_right = np.arange(nrows)*y_spacing

    # make initial grid
    layout_x_left = np.array([x for x in xlocs_left for y in ylocs_left])
    layout_y_left = np.array([y for x in xlocs_left for y in ylocs_left])

    layout_x_right = np.array([x for x in xlocs_right for y in ylocs_right])
    layout_y_right = np.array([y for x in xlocs_right for y in ylocs_right])

    layout_x = np.append(layout_x_left,layout_x_right)
    layout_y = np.append(layout_y_left,layout_y_right)

    return layout_x, layout_y


def evaluate_plant_scenario2_shipping(x):
    global floris_model
    global wd
    global ws
    global wf
    global poly
    global poly_line
    global center_x
    global center_y
    global nrows
    global nturbs

    rotation = x["rotation"]
    shear = x["shear"]
    x_spacing = x["x_spacing"]
    y_spacing = x["y_spacing"]

    base_x, base_y = create_initial_grid_scenario2_shipping(nrows,nturbs,x_spacing,y_spacing)
    shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)

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

    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))

    AEP = floris_model.get_farm_AEP_parallel(wd, ws, wf)

    funcs = {}
    fail = False
    funcs["obj"] = -AEP/1E12
    
    return funcs, fail


def create_initial_grid_scenario3(nrows,nturbs,x_spacing,y_spacing):

    ncols = int(np.floor(nturbs/nrows))
    extra_turbs = nturbs%(nrows*ncols)

    if extra_turbs != 0:
        ncols += 1

    ncols_left = int(1)
    ncols_right = ncols-ncols_left

    nm = 3704/2
    xlocs_left = np.arange(ncols_left)*x_spacing
    ylocs_left = np.arange(nrows)*y_spacing

    shipping_lane_width = 5*nm
    xlocs_right = np.arange(ncols_right)*x_spacing + max(xlocs_left) + shipping_lane_width
    ylocs_right = np.arange(nrows)*y_spacing

    # make initial grid
    layout_x_left = np.array([x for x in xlocs_left for y in ylocs_left])
    layout_y_left = np.array([y for x in xlocs_left for y in ylocs_left])

    layout_x_right = np.array([x for x in xlocs_right for y in ylocs_right])
    layout_y_right = np.array([y for x in xlocs_right for y in ylocs_right])

    layout_x = np.append(layout_x_left,layout_x_right)
    layout_y = np.append(layout_y_left,layout_y_right)

    return layout_x, layout_y


def evaluate_plant_scenario3(x):
    global floris_model
    global wd
    global ws
    global wf
    global poly
    global poly_line
    global center_x
    global center_y
    global nrows
    global nturbs

    rotation = x["rotation"]
    shear = x["shear"]
    x_spacing = x["x_spacing"]
    y_spacing = x["y_spacing"]

    base_x, base_y = create_initial_grid_scenario3(nrows,nturbs,x_spacing,y_spacing)
    shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)

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

    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))

    AEP = floris_model.get_farm_AEP_parallel(wd, ws, wf)

    
    funcs = {}
    fail = False
    funcs["obj"] = -AEP/1E12
    
    return funcs, fail


if __name__=="__main__":

    global base_x
    global base_y
    global poly
    global poly_line
    global center_x
    global center_y
    global nrows
    global nturbs
    

    # INITIAL SETUP
    start_setup = time.time()
    # print("start setup")
    floris_model = wfct.floris_interface.FlorisInterface("12MW.json")
    floris_model.set_gch(False)
    rotor_diameter = floris_model.floris.farm.turbines[0].rotor_diameter

    wind_rose = rose.WindRose()
    wind_rose.load("138m_data.p")
    # wind_rose.load("150m_data.p")
    wind_rose.df = wind_rose.resample_wind_direction(wind_rose.df)
    wind_rose.df = wind_rose.resample_average_ws_by_wd(wind_rose.df)

    wd = wind_rose.df.wd
    ws = wind_rose.df.ws
    wf = wind_rose.df.freq_val

    pt1 = geometry.Point(6478.376698485294, -6120.629680514268)
    pt2 = geometry.Point(17188.12742169846, -6828.612310049923)
    pt3 = geometry.Point(-6478.3766984852955, 6120.629680514268)
    pt4 = geometry.Point(-17188.12742169846, 6828.612310049923) 
    poly = geometry.Polygon((pt1,pt2,pt3,pt4))
    poly_line = geometry.LineString((pt1,pt2,pt3,pt4))

    center_x = 0
    center_y = 0

    x = {}

    nturbs = 66
    # nturbs = 53


    # scenario 1
    # nrows = 11
    # x["rotation"] = np.deg2rad(-3.7821124525353995)
    # x["shear"] = np.deg2rad(-65.09662556182657)
    # x["x_spacing"] = 10.0*rotor_diameter
    # x["y_spacing"] = 5.291999999999996*rotor_diameter

    # scenario 2
    # nrows = 9
    # x["rotation"] = np.deg2rad(-6.729068764733358)
    # x["shear"] = np.deg2rad(-70.15322054461592)
    # x["x_spacing"] = 6.464320092410436*rotor_diameter
    # x["y_spacing"] = 4.76671180802865*rotor_diameter
    # x["rotation"] = np.deg2rad(-3.7770296857551013)
    # x["shear"] = np.deg2rad(-67.17173026047047)
    # x["x_spacing"] = 5.461833275378126*rotor_diameter
    # x["y_spacing"] = 6.6159702487796395*rotor_diameter

    # # scenario 2-shipping
    # nrows = 6
    # x["rotation"] = np.deg2rad(-28.686929113013832)
    # x["shear"] = np.deg2rad(59.999999999921755)
    # x["x_spacing"] = 9.865444676689298*rotor_diameter
    # x["y_spacing"] = 4.210542046663437*rotor_diameter

    # scenario 3
    nrows = 6
    # x["rotation"] = np.deg2rad(-28.68748321172453)
    # x["shear"] = np.deg2rad(59.99999999999999)
    # x["x_spacing"] = 9.149972349841175*rotor_diameter
    # x["y_spacing"] = 4.1755368179339865*rotor_diameter

    # HIGH
    # x["rotation"] = np.deg2rad(-27.854563044789867)
    # x["shear"] = np.deg2rad(65.92697793954974)
    # x["x_spacing"] = 9.622421635823923*rotor_diameter
    # x["y_spacing"] = 3.5464472434988137*rotor_diameter

    # LOW
    x["rotation"] = np.deg2rad(-28.68557127058594)
    x["shear"] = np.deg2rad(58.00224072346609)
    x["x_spacing"] = 8.07835292930626*rotor_diameter
    x["y_spacing"] = 3.501106354199915*rotor_diameter

    # scenario 4
    # nrows = 8
    # x["x_spacing"] = 6.819930016224533*rotor_diameter
    # x["y_spacing"] = 6.701497979280739*rotor_diameter
    # x["rotation"] = np.deg2rad(2.089427145449624)
    # x["shear"] = np.deg2rad(-59.96520712784198)

    # x["x_spacing"] = 5.538605679009052*rotor_diameter
    # x["y_spacing"] = 7.512108156363803*rotor_diameter
    # x["rotation"] = np.deg2rad(180.68667168066477)
    # x["shear"] = np.deg2rad(-61.92396965686597)

    funcs,_ = evaluate_plant_scenario3(x)
    aep = -funcs["obj"]
    base_aep = 3.7448855686274465
    percent_diff = (aep-base_aep)/base_aep*100.0
    print("AEP: ", aep)
    

    base_x,base_y = create_initial_grid_scenario3(nrows,nturbs,x["x_spacing"],x["y_spacing"])
    shear_x,shear_y = shear_grid_locs(x["shear"],base_x,base_y)
    opt_x,opt_y = rotate_grid_locs(x["rotation"],shear_x,shear_y)

    middle_x = (np.max(opt_x)+np.min(opt_x))/2.0
    middle_y = (np.max(opt_y)+np.min(opt_y))/2.0
    dx = center_x-middle_x
    dy = center_y-middle_y
    opt_x = opt_x + dx
    opt_y = opt_y + dy
    ncols = len(opt_x)/nrows
    additional = len(opt_x)-nturbs
    index = np.arange(additional)+int(nrows*(ncols-1))
    opt_x = np.delete(opt_x,index)
    opt_y = np.delete(opt_y,index)

    spacing = calc_spacing(opt_x,opt_y)
    min_spacing = np.min(spacing)/rotor_diameter
    # print("min_spacing: ", min_spacing)


    print("turbine_x = np.%s"%repr(opt_x))
    print("turbine_y = np.%s"%repr(opt_y))

    plt.figure(figsize=(6,3))
    plot_poly(poly,plt.gca())
    plot_turbines(opt_x,opt_y,rotor_diameter/2.0,"C0")
    plt.axis("equal")
    plt.xlim(-19000,19000)
    # plt.ylim(-100,14000)

    plt.ylabel("y coordinates (m)",fontsize=10)
    plt.xlabel("x coordinates (m)",fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.subplots_adjust(left=0.15,right=0.7,top=0.98,bottom=0.15)

    D = 2000
    plt.text(20000,-5000-D,"Scenario 3 - Low",weight="bold")
    plt.text(20000,-7000-D,"AEP: %s GWh(%s"%(int(aep*1000),np.round(percent_diff,1))+"%)")
    # plt.text(20000,-7000-D,"AEP: %s GWh"%int(aep*1000))
    plt.text(20000,-9000-D,"Min. Spacing: %s D"%np.round(min_spacing,2))
    plt.text(20000,-11000-D,"5 nm fishing area")
    # plt.text(20000,-11000-D,"2 nm shipping lane")
    # plt.text(20000,-11000-D,"2 fewer rows")
    # plt.text(20000,-11000-D,"15 MW turbines")


    ax = plt.gcf().add_axes([0.65, 0.5, 0.4, 0.4], polar=True)
    #  [left, bottom, width, height]

    wind_rose.plot_wind_rose(wd_bins=np.arange(0, 360, 5.0),ax=ax)
    ax.get_legend().remove()
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.savefig("figures/try3_scenario3_low.pdf",transparent=True)
    plt.show()