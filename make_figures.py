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


def create_initial_grid_scenario1(a,b,c):
    nrows = 7
    ncols = 9
    x_spacing = 7*rotor_diameter
    y_spacing = 7*rotor_diameter
    xlocs = np.arange(0,ncols)*x_spacing
    ylocs = np.arange(0,nrows)*y_spacing

    # make initial grid
    layout_x = np.array([x for x in xlocs for y in ylocs])
    layout_y = np.array([y for x in xlocs for y in ylocs])

    # add on extra turbines
    layout_x = np.append(layout_x,np.zeros(3)+x_spacing*ncols)
    layout_y = np.append(layout_y,(nrows-np.array([3,2,1]))*y_spacing)

    return layout_x, layout_y


def evaluate_plant_scenario1(x):
    global floris_model
    global wd
    global ws
    global wf
    global base_x
    global base_y

    rotation = x["rotation"]
    shear = x["shear"]

    base_x,base_y = create_initial_grid_scenario1(0,0,0)
    shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)
    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))

    AEP = floris_model.get_farm_AEP_parallel(wd, ws, wf)

    funcs = {}
    fail = False
    funcs["obj"] = -AEP/1E12
    
    return funcs, fail


def create_initial_grid_scenario2(nrows,x_spacing,y_spacing):

    ncols = int(np.floor(66/nrows))
    extra_turbs = 66%(nrows*ncols)

    xlocs = np.arange(ncols)*x_spacing
    ylocs = np.arange(nrows)*y_spacing

    # make initial grid
    layout_x = np.array([x for x in xlocs for y in ylocs])
    layout_y = np.array([y for x in xlocs for y in ylocs])

    # add on extra turbines
    
    # x_spacing = xlocs[1]-xlocs[0]
    # y_spacing = ylocs[1]-ylocs[0]
    layout_x = np.append(layout_x,np.zeros(extra_turbs)+x_spacing*ncols)
    layout_y = np.append(layout_y,(nrows-np.linspace(extra_turbs,1,extra_turbs))*y_spacing)

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

    rotation = x["rotation"]
    shear = x["shear"]
    x_spacing = x["x_spacing"]
    y_spacing = x["y_spacing"]

    base_x, base_y = create_initial_grid_scenario2(nrows,x_spacing,y_spacing)
    shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)

    middle_x = (np.max(turbine_x)-np.min(turbine_x))/2.0 + np.min(turbine_x)
    middle_y = (np.max(turbine_y)-np.min(turbine_y))/2.0 + np.min(turbine_y)
    dx = center_x-middle_x
    dy = center_y-middle_y
    turbine_x = turbine_x + dx
    turbine_y = turbine_y + dy
    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))

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

    print(np.max(constraint_array))

    funcs = {}
    fail = False
    funcs["obj"] = -AEP/1E12
    funcs["constraint"] = np.max(constraint_array)
    
    return funcs, fail


def create_initial_grid_scenario2_shipping(nrows,x_spacing,y_spacing):

    ncols = int(np.floor(66/nrows))
    extra_turbs = 66%(nrows*ncols)

    ncols_right = ncols/2
    if ncols%2 == 0:
        ncols_left = ncols_right
    else: 
        ncols_left = ncols_right+1

    xlocs_left = np.arange(ncols_left)*x_spacing
    ylocs_left = np.arange(nrows)*y_spacing

    shipping_lane_width = 3704
    xlocs_right = np.arange(ncols_right)*x_spacing + max(xlocs_left) + shipping_lane_width
    ylocs_right = np.arange(nrows)*y_spacing

    # make initial grid
    layout_x_left = np.array([x for x in xlocs_left for y in ylocs_left])
    layout_y_left = np.array([y for x in xlocs_left for y in ylocs_left])

    layout_x_right = np.array([x for x in xlocs_right for y in ylocs_right])
    layout_y_right = np.array([y for x in xlocs_right for y in ylocs_right])

    layout_x = np.append(layout_x_left,layout_x_right)
    layout_y = np.append(layout_y_left,layout_y_right)

    # add on extra turbines
    
    # x_spacing = xlocs[1]-xlocs[0]
    # y_spacing = ylocs[1]-ylocs[0]
    
    layout_x = np.append(layout_x,np.zeros(extra_turbs)+x_spacing*(ncols+1) + shipping_lane_width)
    layout_y = np.append(layout_y,(nrows-np.linspace(extra_turbs,1,extra_turbs))*y_spacing)
    
    # plot_turbines(layout_x,layout_y,rotor_diameter/2.0,color="C0")
    # plt.axis("equal")
    # plt.show()

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

    rotation = x["rotation"]
    shear = x["shear"]
    x_spacing = x["x_spacing"]
    y_spacing = x["y_spacing"]

    base_x, base_y = create_initial_grid_scenario2_shipping(nrows,x_spacing,y_spacing)
    shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)

    middle_x = (np.max(turbine_x)+np.min(turbine_x))/2.0
    middle_y = (np.max(turbine_y)+np.min(turbine_y))/2.0
    dx = center_x-middle_x
    dy = center_y-middle_y
    turbine_x = turbine_x + dx
    turbine_y = turbine_y + dy

    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))

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
    pt5 = geometry.Point(turbine_x[-3],turbine_y[-3])

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

    # print(np.max(constraint_array))

    funcs = {}
    fail = False
    funcs["obj"] = -AEP/1E12
    funcs["constraint"] = np.max(constraint_array)
    
    return funcs, fail


def create_initial_grid_scenario3(nrows,x_spacing,y_spacing):

    ncols = int(np.floor(66/nrows))
    extra_turbs = 66%(nrows*ncols)

    ncols_left= 2
    ncols_right = ncols-ncols_left

    xlocs_left = np.arange(ncols_left)*x_spacing
    ylocs_left = np.arange(nrows)*y_spacing

    nm = 3704/2
    fishing_lane_width = 3*nm
    xlocs_right = np.arange(ncols_right)*x_spacing + max(xlocs_left) + fishing_lane_width
    ylocs_right = np.arange(nrows)*y_spacing

    # make initial grid
    layout_x_left = np.array([x for x in xlocs_left for y in ylocs_left])
    layout_y_left = np.array([y for x in xlocs_left for y in ylocs_left])

    layout_x_right = np.array([x for x in xlocs_right for y in ylocs_right])
    layout_y_right = np.array([y for x in xlocs_right for y in ylocs_right])

    layout_x = np.append(layout_x_left,layout_x_right)
    layout_y = np.append(layout_y_left,layout_y_right)

    # add on extra turbines
    
    # x_spacing = xlocs[1]-xlocs[0]
    # y_spacing = ylocs[1]-ylocs[0]
    
    layout_x = np.append(layout_x,np.zeros(extra_turbs)+np.max(layout_x) + x_spacing)
    layout_y = np.append(layout_y,(nrows-np.linspace(extra_turbs,1,extra_turbs))*y_spacing)
    
    # plot_turbines(layout_x,layout_y,rotor_diameter/2.0,color="C0")
    # plt.axis("equal")
    # plt.show()

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

    rotation = x["rotation"]
    shear = x["shear"]
    x_spacing = x["x_spacing"]
    y_spacing = x["y_spacing"]

    base_x, base_y = create_initial_grid_scenario3(nrows,x_spacing,y_spacing)
    shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
    turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)

    middle_x = (np.max(turbine_x)+np.min(turbine_x))/2.0
    middle_y = (np.max(turbine_y)+np.min(turbine_y))/2.0
    dx = center_x-middle_x
    dy = center_y-middle_y
    turbine_x = turbine_x + dx
    turbine_y = turbine_y + dy

    floris_model.reinitialize_flow_field(layout_array=(turbine_x,turbine_y))

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
    pt5 = geometry.Point(turbine_x[-3],turbine_y[-3])

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

    # print(np.max(constraint_array))

    funcs = {}
    fail = False
    funcs["obj"] = -AEP/1E12
    funcs["constraint"] = np.max(constraint_array)
    
    return funcs, fail


if __name__=="__main__":

    global base_x
    global base_y
    global poly
    global poly_line
    global center_x
    global center_y
    global nrows

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

    x = {}

    # scenario 1
    # nrows = 7
    # x["rotation"] = np.deg2rad(23.295214488660186)
    # x["shear"] = np.deg2rad(6.568697451203501)
    # x["x_spacing"] = 7*rotor_diameter
    # x["y_spacing"] = 0

    # # scenario 2-5
    # nrows = 5
    # x["rotation"] = np.deg2rad(23.167119514337468)
    # x["shear"] = np.deg2rad(0.8081012850092529)
    # x["x_spacing"] = 1031.6093644487323
    # x["y_spacing"] = 2153.919581566806

    # # scenario 2-5s
    # nrows = 5
    # x["rotation"] = np.deg2rad(23.173548046771003)
    # x["shear"] = np.deg2rad(3.5194667158188215)
    # x["x_spacing"] = 689.3447108685198
    # x["y_spacing"] = 2161.0496365982735
    
    # # scenario 2-7s
    # nrows = 7
    # x["rotation"] = np.deg2rad(21.366227378821605)
    # x["shear"] = np.deg2rad(2.7218856331191317)
    # x["x_spacing"] = 905.4514978228887
    # x["y_spacing"] = 1332.5020952234902

    # # scenario 3-5
    # nrows = 5
    # x["rotation"] = np.deg2rad(23.356401404414434)
    # x["shear"] = np.deg2rad(4.1512926154477725)
    # x["x_spacing"] = 648.1922418177163
    # x["y_spacing"] = 2191.394267961296

    # # scenario 3-7
    # nrows = 7
    # x["rotation"] = np.deg2rad(27.50584278319377)
    # x["shear"] = np.deg2rad(7.007438496078262)
    # x["x_spacing"] = 726.7446587743422
    # x["y_spacing"] = 1290.1050179493386

    funcs,_ = evaluate_plant_scenario1(x)
    aep = -funcs["obj"]
    base_aep = 3.551283422816099
    percent_diff = (aep-base_aep)/base_aep*100.0
    min_spacing = x["x_spacing"]/rotor_diameter
    print("AEP: ", aep)
    print("min spacing: ", min_spacing)
    

    base_x,base_y = create_initial_grid_scenario1(nrows,x["x_spacing"],x["y_spacing"])
    shear_x,shear_y = shear_grid_locs(x["shear"],base_x,base_y)
    opt_x,opt_y = rotate_grid_locs(x["rotation"],shear_x,shear_y)

    # middle_x = (np.max(opt_x)+np.min(opt_x))/2.0
    # middle_y = (np.max(opt_y)+np.min(opt_y))/2.0
    # dx = center_x-middle_x
    # dy = center_y-middle_y
    # opt_x = opt_x + dx
    # opt_y = opt_y + dy

    plt.figure(figsize=(6,3))
    plot_poly(poly,plt.gca())
    plot_turbines(opt_x,opt_y,rotor_diameter/2.0,"C0")
    plt.axis("equal")
    plt.xlim(-5000,15000)
    plt.ylim(-100,14000)

    plt.ylabel("y coordinates (m)",fontsize=10)
    plt.xlabel("x coordinates (m)",fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.subplots_adjust(left=0.15,right=0.7,top=0.98,bottom=0.15)

    plt.text(15400,3000,"Scenario 1",weight="bold")
    # plt.text(15400,2000,"AEP: %s GWh(%s"%(int(aep*1000),np.round(percent_diff,1))+"%)")
    plt.text(15400,2000,"AEP: %s GWh"%int(aep*1000))
    plt.text(15400,1000,"Min. Spacing: %s D"%np.round(min_spacing,2))
    # plt.text(15400,-1000,"7 rows - 3 nm\nfishing area")
    # plt.text(15400,-1000,"7 rows - 2 nm\nshipping lane")
    # plt.text(15400,0,"5 rows")


    ax = plt.gcf().add_axes([0.65, 0.5, 0.4, 0.4], polar=True)
    #  [left, bottom, width, height]

    wind_rose.plot_wind_rose(wd_bins=np.arange(0, 360, 5.0),ax=ax)
    ax.get_legend().remove()
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.savefig("figures/scenario1.pdf",transparent=True)
    plt.show()