#%%
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Use Tkinter instead of Qt
import matplotlib.pyplot as plt
from pandas import read_csv
#%%
data = read_csv("data/N2_vr.csv")
z, E = np.array(data["R"]), np.array(data["E"])

plt.plot(z, E)
plt.title(f"N$_2$   Bond Length: {(z[np.argmin(E)]):.2f} a$_0$   Total Energy: {min(E):.2f} E$_h$", fontsize=15)
plt.ylabel("Energy / Hartree", fontsize=15)
plt.xlabel("distance / a$_0$", fontsize=15)
#plt.xlim([1.5, 2.5])
#plt.ylim(-115, -105)
plt.show()

# %%
