import numpy as np
import matplotlib.pyplot as plt


side = 900.0

plt.figure(figsize=(3,3))

nrows = 6
spacing = 180.0
farm_width = (nrows-1)*spacing


nrows = 5
ncols = 5
farm_width = 900.0
farm_height = 1200.0
xlocs = np.linspace(0.0,farm_width,ncols)
ylocs = np.linspace(0.0,farm_height,nrows)

center_x = 500.0
center_y = 400.0
shear = np.deg2rad(30.0)
rotation = np.deg2rad(20.0)

y_spacing = ylocs[1]-ylocs[0]
x_spacing = xlocs[1]-xlocs[0]

max_x = farm_width + (nrows-1)*y_spacing*np.tan(shear)
max_y = (nrows-1)*y_spacing
mean_x = 543.1686658980069
mean_y = 836.2038900585054

for i in range(nrows):
    row_x = np.array([0.0,farm_width])+ float(i)*y_spacing*np.tan(shear)
    row_y = np.array([y_spacing*i,y_spacing*i])
    # rotate
    rotate_x = np.cos(rotation)*row_x - np.sin(rotation)*row_y
    rotate_y = np.sin(rotation)*row_x + np.cos(rotation)*row_y
    # move center of grid
    rotate_x = (rotate_x - mean_x) + center_x
    rotate_y = (rotate_y - mean_y) + center_y

    plt.plot(rotate_x,rotate_y,linewidth=0.5,color="black")


for i in range(ncols):
    col_x = np.array([x_spacing*i,x_spacing*i+farm_height*np.tan(shear)])
    col_y = np.array([0.0,farm_height])
    # rotate
    rotate_x = np.cos(rotation)*col_x - np.sin(rotation)*col_y
    rotate_y = np.sin(rotation)*col_x + np.cos(rotation)*col_y
    # move center of grid
    rotate_x = (rotate_x - mean_x) + center_x
    rotate_y = (rotate_y - mean_y) + center_y

    plt.plot(rotate_x,rotate_y,linewidth=0.5,color="black")

plt.plot([center_x,center_x+500],[center_y,center_y],'--k')

from matplotlib import patches
e1 = patches.Arc((center_x, center_y), 800.0, 800.0,
                 theta1=0.0,theta2=np.rad2deg(rotation),color="C2",linewidth=2)
plt.gca().add_patch(e1)
L = 500.0
plt.plot([-43.16,-43.16-L*np.sin(rotation)],[-436.2,-436.2+L*np.cos(rotation)],'--k')
plt.plot([-43.16,-43.16+L*np.cos(rotation)],[-436.2,-436.2+L*np.sin(rotation)],'--k')
e2 = patches.Arc((-43.16, -436.2), 800.0, 800.0,
                 theta1=90.0+np.rad2deg(rotation)-np.rad2deg(shear),theta2=90.0+np.rad2deg(rotation),
                 color="C3",linewidth=2)
plt.gca().add_patch(e2)

plt.plot([831,1044],[1159,1236],"--",color="C0",linewidth=2)
plt.plot([831,831+y_spacing*np.sin(rotation)],[1159,1159-y_spacing*np.cos(rotation)],'--',color="C1",linewidth=2)

plt.axis("off")
plt.xlim(-500.0,1100)
plt.ylim(-500.0,1300)

plt.plot([-43.16,-43.16-farm_height*np.sin(rotation)],[-436.2,-436.2+farm_height*np.cos(rotation)],'--k')
plt.plot([-43.16,-43.16+farm_width*np.cos(rotation)],[-436.2,-436.2+farm_width*np.sin(rotation)],'--k')

plt.text(center_x+800,center_y+75,"rotation",horizontalalignment="center",verticalalignment="center",color="C2")
plt.text(-500,-20,"shear",horizontalalignment="center",verticalalignment="center",color="C3")
plt.text(995,1300,"x spacing",horizontalalignment="center",verticalalignment="center",color="C0")
plt.text(1200,1000,"y spacing",horizontalalignment="center",verticalalignment="center",color="C1")


plt.axis("equal")
plt.tight_layout()
plt.savefig("grid_variables.pdf",transparent=True)
plt.show()