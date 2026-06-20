import matplotlib.pyplot as plt

# Data
z = [0.1000, 4.0500, 8.0000, 11.9500, 15.9000]
e_3d = [-0.1702, -0.1281, -0.2335, -0.1281, -0.1702]
e_dipole = [0.0123, 0.0031, 0.0000, 0.0031, 0.0123]
e_recip = [0.0989, 0.0140, -0.0023, 0.0140, 0.0989]
sum_e = [-0.0590, -0.1110, -0.2358, -0.1110, -0.0590]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(z, e_3d, marker='o', label='E_3d')
plt.plot(z, e_dipole, marker='s', label='E_dipole')
plt.plot(z, e_recip, marker='^', label='E_recip')
plt.plot(z, sum_e, marker='x', linestyle='--', label='Sum')

plt.xlabel('Z value')
plt.ylabel('Energy')
plt.title('Energy Components vs Z')
plt.legend()
plt.grid(True)
plt.show()