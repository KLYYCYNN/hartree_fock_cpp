import numpy as np
import matplotlib.pyplot as plt

# load density matrix slice
rho = np.loadtxt("/home/jc6224/hf/data/test1/homo_lumo/2dlumo.csv", delimiter=",")

# plot
plt.figure(figsize=(6,6))

plt.imshow(
    rho,
    origin="lower",
    cmap="inferno"
)

plt.tight_layout()
plt.show()