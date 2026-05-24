import json
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

# load density matrix slice
rho = np.loadtxt("/home/jc6224/hf_cpp/data/H2O/homo_lumo1/lumo2d.csv", delimiter=",")
params = read_csv("/home/jc6224/hf_cpp/data/H2O/homo_lumo1/square.csv")



try:
    # Open and read the JSON file
    with open("/home/jc6224/hf_cpp/data/H2O/homo_lumo1/square.json", 'r') as file:
        data = json.load(file)
    
    # Get the value and convert it to a float
    L = float(data['side_length'])

except FileNotFoundError:
    print("Error: The file 'square.json' was not found.")
except KeyError:
    print("Error: The key 'side_length' is missing from the JSON file.")
except ValueError:
    print("Error: The value of 'side_length' cannot be converted to a float.")


u_vec = np.array(params["hside_direc"])
v_vec = np.array(params["vside_direc"])

def vec_label(vec):
    x, y, z = vec
    return f"({x:.2f}, {y:.2f}, {z:.2f})"



plt.imshow(rho, extent=[-L/2, L/2, -L/2, L/2], origin="lower")

# horizontal slice-axis arrow
plt.arrow(
    -0.4*L, -0.45*L,
    0.25*L, 0,
    head_width=0.05*L,
    length_includes_head=True
)

plt.text(
    -0.12*L, -0.45*L,
    f"u = {vec_label(u_vec)}",
    va="center"
)

# vertical slice-axis arrow
plt.arrow(
    -0.45*L, -0.4*L,
    0, 0.25*L,
    head_width=0.05*L,
    length_includes_head=True
)

plt.text(
    -0.45*L, -0.12*L,
    f"v = {vec_label(v_vec)}",
    ha="center",
    rotation=90
)


plt.xlabel("slice coordinate along u / a$_0$")
plt.ylabel("slice coordinate along v / a$_0$")
plt.colorbar(label=r"$\psi_i$")
plt.show()