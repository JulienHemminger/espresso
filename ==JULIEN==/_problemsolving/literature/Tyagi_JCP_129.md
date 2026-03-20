### II. SYSTEM

The system setup is shown in Fig. 1. We consider a charge neutral set of $N$ charges in a medium characterized by a dielectric constant $\epsilon_m$ that is sandwiched from above by a medium of dielectric constant $\epsilon_t$ and from below by a medium of dielectric constant $\epsilon_b$. We thus have two interfaces characterizing the problem. The case of a single interface is included since one can remove an interface by choosing $\epsilon_t = \epsilon_m$ for example. It is assumed that the interfaces act as barriers and thus charged particles are not allowed to migrate across them. For the case where only a single interface exists, one can assume the presence of an invisible wall on the other side that prevents the particles from escaping to infinity. Such an impenetrable wall can be described by a repulsive Lennard-Jones interaction, for example.

The two dielectric interfaces can be treated by using the method of image charges. A single dielectric interface may be replaced by a single image charge for every real charge in the system. However, the situation becomes more complicated when there are two parallel dielectric interfaces such as the case we are considering here. In this case, one ends up with an infinite number of image charges due to the repeated reflections of image charges under the two parallel dielectric interfaces facing each other, as depicted in Fig. 1. Nevertheless, in both cases, the electrostatic energy can be determined by summing over all interactions of the real charges with both real and image charges.

To make things concrete, let us consider two interfaces located at $z=0$ and at $z=l_z$. A set $q_i \in L$ of charges are confined within the central dielectric layer. For each charge $q_i$, we have an infinite series of image charges along the $z$-direction. For example, a charge $q_i$ at a position $z_i$ gives rise to an image charge $\Delta_b q_i$ at $z = -z_i$ in the lower dielectric layer and an image charge $\Delta_t q_i$ at $z = 2l_z - z_i$ in the upper dielectric layer. Here, the prefactors $\Delta_b$ and $\Delta_t$ are defined as

$$\Delta_b = \frac{\epsilon_m - \epsilon_b}{\epsilon_m + \epsilon_b}, \quad \Delta_t = \frac{\epsilon_m - \epsilon_t}{\epsilon_m + \epsilon_t} \tag{2.1}$$

In addition, other image charges appear due to repeated images of the image charges themselves under the two planar dielectric interfaces. For each one of the lower and upper dielectric layers, one can separate the image charges into two sequences. It is easy to see that one would have the following two sequences of charges in the lower dielectric layer (compare also Fig. 1):

**Charge:** $$q_i \Delta_b, \quad q_i \Delta \Delta_b, \quad q_i \Delta^2 \Delta_b, \dots$$
**Position:** $$-z_i, \quad -(2l_z + z_i), \quad -(4l_z + z_i), \dots \tag{2.2}$$

and

**Charge:** $$q_i \Delta, \quad q_i \Delta^2, \quad q_i \Delta^3, \dots$$
**Position:** $$-(2l_z - z_i), \quad -(4l_z - z_i), \quad -(6l_z - z_i), \dots \tag{2.3}$$

where $$\Delta = \Delta_b \Delta_t$$. Similarly, two sequences of charges can be derived for the upper dielectric domain:

**Charge:** $$q_i \Delta_t, \quad q_i \Delta \Delta_t, \quad q_i \Delta^2 \Delta_t, \dots$$
**Position:** $$(2l_z - z_i), \quad (4l_z - z_i), \quad (6l_z - z_i), \dots \tag{2.4}$$

and

**Charge:** $$q_i \Delta, \quad q_i \Delta^2, \quad q_i \Delta^3, \dots$$
**Position:** $$(2l_z + z_i), \quad (4l_z + z_i), \quad (6l_z + z_i), \dots \tag{2.5}$$

The electrostatic energy of the system is then obtained by summing up the interactions of the charges in $L$ with the charges in $L$ and with all image charges. Since $\Delta \le 1$ this sum always exists.