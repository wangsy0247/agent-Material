# A deep neural network interatomic potential for studying thermal conductivity of β-Ga2O3.pdf

> MinerU Flash 批处理解析  |  共 7 页  |  1 个切片

---

## 第 1–7 页

RESEARCH ARTICLE | OCTOBER 14 2020

# A deep neural network interatomic potential for studying thermal conductivity of β-Ga O 

Special Collection: Ultrawide Bandgap Semiconductors

Ruiyang Li ; Zeyu Liu; Andrew Rohskopf; Kiarash Gordiz; Asegun Henry; Eungkyu Lee  ; Tengfei Luo 

Check for updates

Appl. Phys. Lett. 117, 152102 (2020) https://doi.org/10.1063/5.0025051

<!-- image-->

## Articles You May Be Interested In

A moment tensor potential for lattice thermal conductivity calculations of α and β phases of $\mathsf { G a } _ { 2 } \mathsf { O } _ { 3 }$

J. Appl. Phys. (May 2024)

Using Green-Kubo modal analysis (GKMA) and interface conductance modal analysis (ICMA) to study phonon transport with molecular dynamics

J. Appl. Phys. (February 2019)

Thermal transport properties of monolayer GeS and SnS: A comparative study based on machine learning and SW interatomic potential models

AIP Advances (August 2022)

<!-- image-->

## AIP Advances

## Why Publish With Us?

<!-- image-->

21DAYS averagetime to1st decision

<!-- image-->

OVER 4 MILLION views in the last year

<!-- image-->

INCLUSIVE scope

Learn More

# A deep neural network interatomic potential for studying thermal conductivity of $\beta \mathrm { - } \mathsf { C a } _ { 2 } \mathsf { O } _ { 3 }$

Cite as: Appl. Phys. Lett. 117, 152102 (2020); doi: 10.1063/5.0025051 Submitted: 12 August 2020 . Accepted: 25 September 2020 . Published Online: 14 October 2020

<!-- image-->

Ruiyang Li,1 Zeyu Liu,1 Andrew Rohskopf,2 Kiarash Gordiz,2 Asegun Henry,2 Eungkyu Lee,1,3,a) and Tengfei Luo1,4,5,a)

## AFFILIATIONS

1 Department of Aerospace and Mechanical Engineering, University of Notre Dame, Notre Dame, Indiana 46556, USA 2 Department of Mechanical Engineering, Massachusetts Institute of Technology, Cambridge, Massachusetts 02139, USA 3 Department of Mechanical System Engineering, Kumoh National Institute of Technology, 61 Daehak-ro, Gumi-si, Gyeongbuk 39177, South Korea

4 Department of Chemical and Biomolecular Engineering, University of Notre Dame, Notre Dame, Indiana 46556, USA   
5 Center for Sustainable Energy of Notre Dame (ND Energy), University of Notre Dame, Notre Dame, Indiana 46556, USA

Note: This paper is part of the Special Topic on Ultrawide Bandgap Semiconductors.

a)Authors to whom correspondence should be addressed: elee@kumoh.ac.kr and tluo@nd.edu

## ABSTRACT

$\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ is a wide-bandgap semiconductor of significant technological importance for electronics, but its low thermal conductivity is an impeding factor for its applications. In this work, an interatomic potential is developed for $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ based on a deep neural network model to predict the thermal conductivity and phonon transport properties. Our potential is trained by the ab initio energy surface and atomic forces, which reproduces phonon dispersion in good agreement with first-principles calculations. We are able to use molecular dynamics (MD) simulations to predict the anisotropic thermal conductivity of $\mathrm { \bf \ddot { \beta } } { \bf - } { \bf G a } _ { 2 } { \bf O } _ { 3 }$ with this potential, and the calculated thermal conductivity values agree well with experimental results from 200 to 500 K. Green–Kubo modal analysis is performed to quantify the contributions of different phonon modes to the thermal transport, showing that optical phonon modes play a critical role in the thermal transport. This work provides a high-fidelity machine learning-based potential for MD simulation of $\beta { \mathrm { - G a } } _ { 2 } { \mathrm { O } } _ { 3 }$ and serves as a good example of exploring thermal transport physics of complex semiconductor materials.

Published under license by AIP Publishing. https://doi.org/10.1063/5.0025051

As a wide-bandgap transparent oxide semiconductor, $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ has attracted much scientific and technological attention.1 Due to its large critical electric field and relatively inexpensive growth method,2 $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ is a promising candidate for applications in power electronic devices,3 solar-blind UV photodetectors,4 and gas sensors.5 In particular, for high-power and high-voltage switching devices (e.g., transistors), the performance and reliability can be significantly affected by the high temperature induced by localized Joule heating in the channel of oxide semiconductors.6,7 An understanding of the thermal properties of $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ is, therefore, critical for better thermal management and device packaging design.

A few experimental studies have reported the thermal conductivity values of $\beta \mathrm { \dot { - } G a _ { 2 } O _ { 3 } } . ^ { 8 - 1 1 }$ It is shown that the measured thermal conductivity (j) of $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ is anisotropic due to the monoclinic lattice structure.12 Recent computational studies using the first-principles Boltzmann transport equation (BTE) method,13,14 however, failed to predict j values in good agreement with the experimental values, which is likely due to factors like the difficulties in obtaining converged j with respect to the grid size and the broadening coefficient for screening the scattering phase space. In addition, given the large conventional cell of $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ containing 20 atoms, the first-principles calculations are highly time-consuming especially when extensive convergence tasks need to be performed. Ab initio molecular dynamics (AIMD) can also accurately predict thermal conductivity,15 but simulations of $\beta { \mathrm { - G a } } _ { 2 } { \mathrm { O } } _ { 3 }$ will be severely limited by the system size (a few hundred atoms) and timescales (tens of picoseconds). Molecular dynamics (MD) simulation using empirical potentials enables us to efficiently predict properties of both bulk materials and nanostructures, over a wider range of system sizes (up to hundreds of nanometers)16 and timescales (up to tens of nanoseconds),17 even though the prediction accuracy highly depends on the fidelity of the potentials. However, such accurate potentials do not exist for $\beta \mathrm { - } \mathrm { G a } _ { 2 } \mathrm { O } _ { 3 } ,$ and developing them is usually a challenging task, which commonly involves fitting pre-selected functional forms with the ab initio energy surface.18 Depending on the specific problem of interest, the fixed functional form of empirical potentials can limit the applicability of the fitted potential, especially when higher order terms (e.g., force and anharmonicity) are important. Studying thermal transport is one such problem.

Thanks to the recent advances in machine learning (ML)-based techniques,19 one can reconstruct the ab initio potential energy surface using certain ML models, such as artificial neural networks20 and Gaussian process regression21 with very high accuracy. Such ML models can be used as reliable fitting tools to develop interatomic potentials that precisely map the atomic configurations to the corresponding ab initio energies and forces. Several different ML potentials have been proposed to predict thermal properties of crystalline solids such as Si,22 $\mathrm { Z r } , ^ { 2 3 }$ graphene,24 a nd $\mathrm { M o S } _ { 2 } . ^ { \bar { 2 } 5 }$ Recent studies proved that even materials with more complex structures can be well simulated using ML potentials,26,27 and a unified potential has been developed to model different phases of Si and, at the same time, predict thermal conductivity values accurately.28 It has also been proved that long-range interactions are important for the correct description of thermal vibrations in covalent materials, but two-body empirical potentials using radial descriptors cannot properly capture the forces and interatomic force constants due to the lack of flexibility.29 Since ML potentials can provide such flexibility and no accurate potential exists for $\beta \mathrm { - G a } _ { 2 } \mathrm { O } _ { 3 } ,$ a ML potential for $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ will be highly impactful to the community.

In this work, we develop a neural network potential (NNP) to predict the thermal conductivity of $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ with MD simulations. The potential is obtained by training a deep neural network (DNN) model with the ab initio data of a $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ supercell. Accurate j values can be achieved using the NNP, which agree well with the experimental results at different temperatures. Normal mode analysis has been performed to quantify the contributions to thermal transport from different phonon modes.

In order to build the NNP for $\beta { \mathrm { - G a } } _ { 2 } { \mathrm { O } } _ { 3 } ,$ we first need to generate a reference dataset for the training process. AIMD simulations30 are conducted to collect the needed data (potential energy, atomic forces, atomic coordinates, and supercell lattice vectors) of $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ . We perform AIMD simulations at different finite temperatures, along with static self-consistent density functional theory (DFT) calculations of randomly displaced structures at 0 K to diversify the sampled phase space. As for the AIMD simulations at finite temperatures, snapshots are collected from different molecular trajectories in the isobaric– isothermal ensemble with temperature ranging from 50 to 600 K at zero pressure. The system is first equilibrated for 1.0 ps with a time step of 1.0 fs, and then the snapshots are collected every four steps during production runs for 5.0 ps at different temperatures. The ab initio calculations are carried out using the QUICKSTEP algorithm implemented in the CP2K package, which is based on the Gaussian and plane wave (GPW) approaches.31,32 We use the triple-f valence plus polarization (TZVP) basis sets and the Goedecker–Teter–Hutter (GTH) pseudopotentials33 to describe core–valence interactions. The plane wave cutoff is set to be 800 Ry, and the Brillouin zone is sampled at the C point. All the calculations are performed for a relaxed monoclinic $2 \times 4 \times 2 \ \beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ supercell, with the lattice parameters $a = 1 2 . 3 7 6 \mathring \mathrm { A } ,$ b ¼ 3.084 ${ \overset { \circ } { \operatorname { A } } } , c = 5 . 8 9 3 { \overset { \circ } { \operatorname { A } } } .$ and $\beta = 1 0 3 . 8 3 ^ { \circ } \ [ \mathrm { F i g . \ 1 ( a ) } ]$ , in good agreement with the experimental results.34 A total of 9200 ab initio snapshots are generated for the training process.

An end-to-end DNN-based potential scheme is used to obtain the NNP for $\beta { \mathrm { - G a } } _ { 2 } { \mathrm { O } } _ { 3 } .$ This MD potential scheme, namely, the Smooth Edition of Deep Potential (DeePot-SE) developed by Zhang $e t a l . , ^ { 3 5 }$ represents the lattice potential energy as a sum of atomic energies, which can be determined by the local environment of each atom. Within DeePot-SE, all the natural symmetries are preserved by mapping the local environments to an embedded feature space. More details about this scheme and related models can be found in the literature.36,37 In this study, our DNN model is composed of an embedding network and a fitting network, and the only inputs are the chemical species and atomic coordinates. We have used a two-layer embedding network to map the original inputs to symmetrypreserving components. The fitting network is a three-layer feedforward neural network containing 160 nodes in each layer, which maps the embedded features to the atomic energies. The DNN model is trained with the ab initio dataset to minimize the loss function, which is a weighted sum of mean square errors in energy and force. The optimization is solved by TensorFlow’s implementation of the Adam stochastic gradient descent method.38 The DNN structures and training hyper-parameters are also optimized to minimize the prediction errors. All training data and potential files are included in Ref. 39. Then, we use the LAMMPS package40 with this established NNP to perform MD simulations of $\beta { \mathrm { - G a } } _ { 2 } { \mathrm { O } } _ { 3 }$

(a)  
<!-- image-->

(b)  
<!-- image-->  
FIG. 1. (a) A conventional unit cell of the $\beta \mathrm { - } G a _ { 2 } 0 _ { 3 }$ crystal and three crystallographic directions. The green and red spheres represent gallium atoms and oxygen atoms, respectively. (b) Standardized potential energies and atomic forces from MD simulations using NNP and ab initio calculations. Energies and forces are standardized by subtracting the average (l) from them and then divided by the standard deviation (r). Both quantities are plotted within 67r.

The fitting accuracy can be verified by the agreement between NNP predictions and the benchmark AIMD results. We compare the NNP-predicted energies and forces for a separate testing dataset containing 600 AIMD snapshots that are randomly selected and not included in the training dataset. It is found that our NNP can provide very similar results compared to the testing data at different temperatures, with the root mean square errors of 0.0005 eV per atom for the potential energy and 0.078 eV/A˚ for the atomic forces. The mean percent errors in energy and force are 0.0003% and 9.1%, respectively. For comparison, it has been reported that popular empirical potentials such as Tersoff41 and Stillinger–Weber42 potential for crystalline Si result in 35% and 210% error in forces, respectively.43 Figure 1(b) shows the comparison of AIMD energies and forces and NNPpredicted quantities, indicating that the NNP accurately reproduces the ab initio energies and forces.

To evaluate the accuracy of harmonic force constants, we calculate the phonon dispersion relation along selected high symmetry paths. The harmonic force constants are computed by fitting displacement–force data using the Alamode program,44 where symmetryirreducible displacements of 0.01 A˚ are imposed and atomic forces are calculated with this NNP. As shown in Fig. 2(a), the phonon dispersion using NNP agrees well with that from the first-principles calculation by Santia et al., 13 while existing empirical potentials fail to capture that.45,46 Considering the low symmetry and high anisotropy of $\beta \mathrm { - G a } _ { 2 } \mathrm { O } _ { 3 } ,$ it is reasonable to see that the phonon dispersion profile is much more complex than that of many other wide-bandgap materials, such as GaN and ZnO.47 At the same time, it is understandable why it has been very difficult to achieve an excellent agreement in phonon dispersion using a fitted empirical potential composed of simple functions. However, because of the versatility arising from high complexity of the DNN, harmonic force constants can be reproduced accurately with NNP, which are important for the prediction of thermal conductivity.

Equilibrium MD (EMD) simulations are preformed to evaluate the thermal conductivity using the Green–Kubo approach.48–50 The Green–Kubo formula relates the instantaneous fluctuations of heat current, in terms of autocorrelation function, to thermal conductivity as

$$
\kappa _ { \alpha } = \frac { 1 } { k _ { B } T ^ { 2 } V } \int _ { 0 } ^ { \tau _ { 0 } } \big \langle J _ { \alpha } ( 0 ) J _ { \alpha } ( t ) \big \rangle d t .\tag{1}
$$

Here, $\kappa _ { \alpha }$ denotes the thermal conductivity in the a direction, $k _ { B }$ is the Boltzmann constant, V is the volume of the simulation cell, $T$ is the temperature, $\tau _ { O }$ is the integral upper limit, which should be infinite theoretically, t represents the correlation time, $\langle \rangle$ denotes the ensemble average, and $J _ { \alpha }$ is the component of heat current J in the a direction, which is expressed as51

(a)  
<!-- image-->  
(b)

<!-- image-->  
FIG. 2. (a) Phonon dispersion relation of $\beta \mathrm { - } \mathsf { G a } _ { 2 } \mathsf { O } _ { 3 }$ along selected high symmetry paths using NNP. The dispersion relation agrees well with the phonon dispersion from DFT calculations.13 Only the dispersion along the $\Gamma  \dot { \mathsf { X } } _ { 1 }$ path in Fig. S2 (Ref. 13) is shown for comparison. (b) Phonon dispersion along three crystallographic directions, denoted by $\Gamma  \dot { \mathsf { A } } _ { 1 }$ ([100]), $\Gamma  \mathsf { A } _ { 2 }$ ([010]), and $\Gamma \dot { \bf \Phi } \to \mathbb { A } _ { 3 }$ ([001]).

$$
J = \sum _ { i } { \nu _ { i } } E _ { i } + \sum _ { i } { \sum _ { j \ne i } { { r _ { i j } } } } \left( { \frac { \partial U _ { j } } { \partial { { r _ { i j } } } } } \cdot { { \nu _ { i } } } \right) ,\tag{2}
$$

where $E _ { i } , \nu _ { i } , U _ { j } ,$ and $\boldsymbol { r } _ { i j }$ are the site energy, atomic velocity, potential energy, and distance between atoms i and $j ,$ respectively. Conventionally, one defines $\textstyle \sum _ { i }$ viEi as the kinetic part, which is found to barely contribute to the thermal conductivity for solids and can be simply neglected.52 We, thus, only focus on the potential part $\begin{array} { r } { \sum _ { i } \sum _ { j \neq i } r _ { i j } \Big ( \frac { \partial U _ { j } } { \partial r _ { i j } } \cdot \pmb { \nu } _ { i } \Big ) } \end{array}$ for the heat current in this work.

We perform EMD simulations on a bulk $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ system consisting of 4 - 13 - 7 conventional unit cells consisting of 7280 atoms (at least 4 nm long in each crystallographic direction), which is usually not feasible to simulate using first-principles methods. Periodic boundary conditions are applied in all three directions, and the isobaric-isothermal ensemble is first used to equilibrate the system at zero pressure for 200 ps with a time step of 1.0 fs. Then, the simulations are switched to the microcanonical ensemble and run for another 2 ns to collect the heat current data. To suppress the statistical uncertainty, 15 independent MD runs with different initial velocity distributions are performed to obtain the average j values at 200, 300, 400, and 500 K. As mentioned in Ref. 28, MD simulations using NNP are usually slower than those using empirical potentials. With the potential in this work, it takes around 6 days to simulate a 7280-atom system for 2 ns using a single Intel Xeon Silver 4110 CPU core and a NVIDIA TITAN Xp GPU, which suggests that our calculations of thermal conductivity are in a reasonable computing time range.

Figure 3(a) shows the average integral of heat current autocorrelation functions along three crystallographic directions at 300 K. The inset shows the thermal conductivity values for various simulation cell sizes, with the 7280-atom system giving results within the error bar of the case for a 5040-atom system, which are considered converged with respect to the domain size. We have observed the anisotropy in thermal conductivity, with the calculated values at 300 K to be $\kappa _ { [ 1 0 0 ] } = 1 0 . 6 8 , \kappa _ { [ 0 1 0 ] } = 2 0 . 7 8 .$ , and $\kappa _ { [ 0 0 1 ] } = 1 2 . 6 1 \mathrm { W } \mathrm { m } ^ { - 1 } \mathrm { K } ^ { - 1 } ,$ . The temperature dependence of j is plotted in Fig. 3(b) along with the results from time-domain thermoreflectance (TDTR) measurements.8,11 Predictions from first-principles BTE calculations are also included for comparison.13,14 The calculated j values using NNP show good agreement with experimentally observed trends, with the highest j along the [010] direction. However, for the [100] and [001] directions, both NNP predicted results and experimental results are lower than the first-principles results. One of the reasons could be that MD simulations with NNP naturally include higher order anharmonicities, whereas the BTE method truncates the anharmonic force constants to the third order.53 Moreover, another observation is that the BTEcalculated j values in the [010] and [001] directions are essentially the same, which contradicts with both experimental measurements8,11 and our MD calculations. We, thus, cast our doubts on the BTE results,13 which are challenging to obtain accurately given the complexity of the $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ unit cell and uncertainties in convergence with respect to different parameters in such calculations as previously mentioned. At high temperatures (>400 K), our calculated j values in all directions agree quantitatively with experimental measurements. At lower temperatures (e.g., 200 K), the MD results are lower than experiments, which is likely due to the size effect of the simulation domain and the overestimation of the scattering rates that cutoff the contributions from long wavelength phonons, which are more important at lower temperatures due to their increased mean free path. Here, we have not accounted for any quantum effect in our calculations, and there can be two competing quantum effects: classical statistics gives larger modal heat capacities at high frequencies but smaller phonon scattering times at low frequencies compared to quantum statistics.54 Since the quantum correction is still controversial in accounting for quantum effects, we do not include quantum correction in this work. However, for practical applications, $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ thermal conductivity above room temperature is more important, where our NNP-EMD is able to yield good results.

(a)  
<!-- image-->

(b)  
<!-- image-->  
FIG. 3. (a) Thermal conductivity of $\beta \mathrm { - } G a _ { 2 } 0 _ { 3 }$ as a function of correlation time using the Green–Kubo method with NNP. These curves show the thermal conductivity along different crystal orientations at 300 K, each of which is the average of 15 independent simulations with different initial velocity distributions. The inset shows the simulation domain size-dependent thermal conductivity at 300 K. (b) Temperature-dependent thermal conductivity calculated from NNP-EMD for different crystal directions (solid symbols), compared with the reported experimental values,8,1 (cross symbols) and first-principles predictions based on DFT force constants13,14 (solid lines and dashed lines).

To further study the phonon transport physics, we calculate the contribution of different phonon modes to the anisotropic thermal conductivity using Green–Kubo modal analysis (GKMA).55,56 The GKMA method combines the lattice dynamics with the Green–Kubo formula such that the thermal conductivity becomes a direct summation of modal contributions. For a system with N atoms vibrating around their equilibrium positions, one can obtain 3N collective modes of vibration by lattice dynamics calculations on the entire supercell. Then, the MD trajectory can be projected onto the modes by relating the velocity of each atom j to the normal mode velocity ${ \dot { X } } _ { n }$ for mode n,

$$
\pmb { \nu } _ { j } = \dot { \pmb { x } } _ { j } = \sum _ { n } \dot { \pmb { x } } _ { j , n } = \frac { 1 } { \sqrt { m _ { j } } } \sum _ { n } \pmb { e } _ { j , \mathrm { n } } \dot { X } _ { n } ,\tag{3}
$$

where $m _ { j }$ is the mass of atom j and $e _ { j , n }$ is the eigenvector that gives the magnitude and direction of motion for atom j in mode n. The normal mode velocity can be obtained by reverse transformation,

$$
\dot { X } _ { n } = \sum _ { j } \sqrt { m _ { j } } \pmb { e } _ { j , n } ^ { * } \cdot \dot { \pmb { x } } _ { j } ,\tag{4}
$$

where  represents the complex conjugate. Neglecting the kinetic part, we can write the heat current in Eq. (2) as a sum of individual modal contributions,

$$
J = \sum _ { n } ^ { 3 N } J _ { n } = \sum _ { n } ^ { 3 N } \sum _ { i } \sum _ { j \neq i } \boldsymbol { r } _ { i j } \left( \frac { \partial U _ { j } } { \partial \boldsymbol { r } _ { i j } } \cdot \frac { 1 } { \sqrt { m _ { i } } } \boldsymbol { e } _ { i , n } \dot { \boldsymbol { X } } _ { n } \right) .\tag{5}
$$

By substituting $\operatorname { E q . } \left( 5 \right)$ into Eq. (1), we have the contribution of mode n in thermal conductivity

$$
\kappa _ { \alpha , n } = \frac { 1 } { k _ { B } T ^ { 2 } V } \int _ { 0 } ^ { \tau _ { 0 } } \big \langle J _ { \alpha , n } ( 0 ) J _ { \alpha } ( t ) \big \rangle d t .\tag{6}
$$

Using the modes obtained from the finite difference method, we perform a modal analysis based on the GKMA method for $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ at 300 K. The accumulation of thermal conductivity with respect to mode frequency is shown in Fig. 4. Our GKMA results using NNP show that around 80% of j in three crystal directions are contributed by the phonon modes below 10 THz. It is also evident that the optical phonons with frequency above 5 THz play a critical role in the thermal transport, especially in the [010] direction. At 300 K, optical phonons with frequency above 5 THz contribute over 45% of the total j[010]. The high optical phonon contribution may result from the hybridization of low energy optical modes with the longitudinal acoustic mode.13,14 Furthermore, compared to [100] and [001] directions with relatively flat optical phonon branches [Fig. 2(b)], the [010] direction shows larger phonon group velocities, which can lead to a higher j contributed by optical phonons.

In summary, we have developed an interatomic potential for $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ using a deep neural network model. This NNP is trained with ab initio energies and forces such that it can reproduce the phonon dispersion in very good agreement with the first-principles results. Using MD simulations with this NNP, we predict the lattice thermal conductivity in three crystallographic directions over a range of temperatures from 200 to 500 K. The thermal conductivity of $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ is found to exhibit large anisotropy, and our predictions agree well with experimental results in the literature especially above room temperature. Furthermore, the modal analysis using the GKMA method reveals that optical phonon modes contribute significantly to the overall thermal conductivity at 300 K, especially in the [010] direction. These results demonstrate the accuracy of our NNP in predicting thermal properties of $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ . Since the accurate prediction of thermal conductivity implies the accuracy of our NNP in both second and third order force constants, this potential can be a foundation for MD simulations of other $\beta { \mathrm { - G a } } _ { 2 } \mathrm { O } _ { 3 }$ properties, such as mechanical and thermo-mechanical properties. In addition to $\beta { \mathrm { - G a } } _ { 2 } { \mathrm { O } } _ { 3 } ,$ more NNPs can be developed in the same vein to study thermal transport in other complex semiconductor materials.

<!-- image-->  
FIG. 4. Accumulated thermal conductivity along three crystallographic directions with respect to mode frequency at 300 K.

The authors would like to thank ONR MURI (No. N00014-18-1- 2429) for the financial support. The simulations are supported by the Notre Dame Center for Research Computing and NSF through the eXtreme Science and Engineering Discovery Environment (XSEDE) computing resources provided by Texas Advanced Computing Center (TACC) Stampede II under Grant No. TG-CTS100078.

## DATA AVAILABILITY

The data that support the findings of this study are openly available at Ref. 39.

## REFERENCES

1 F. Shan, G. Liu, W. Lee, G. Lee, I. Kim, and B. Shin, J. Appl. Phys. 98, 023504 (2005).

2 S. Pearton, J. Yang, P. H. Cary IV, F. Ren, J. Kim, M. J. Tadjer, and M. A. Mastro, Appl. Phys. Rev. 5, 011301 (2018).

3 M. Higashiwaki, A. Kuramata, H. Murakami, and Y. Kumagai, J. Phys. D 50, 333002 (2017).

4 A. Singh Pratiyush, S. Krishnamoorthy, S. Vishnu Solanke, Z. Xia, R. Muralidharan, S. Rajan, and D. N. Nath, Appl. Phys. Lett. 110, 221107 (2017).

5 M. Fleischer and H. Meixner, Sens. Actuators, B 6, 257 (1992).

6 H. Zhou, K. Maize, G. Qiu, A. Shakouri, and P. D. Ye, Appl. Phys. Lett. 111, 092102 (2017).

7 Z. Cheng, V. D. Wheeler, T. Bai, J. Shi, M. J. Tadjer, T. Feygelson, K. D. Hobart, M. S. Goorsky, and S. Graham, Appl. Phys. Lett. 116, 062105 (2020).

8 Z. Guo, A. Verma, X. Wu, F. Sun, A. Hickman, T. Masui, A. Kuramata, M. Higashiwaki, D. Jena, and T. Luo, Appl. Phys. Lett. 106, 111909 (2015).

9 Z. Galazka, K. Irmscher, R. Uecker, R. Bertram, M. Pietsch, A. Kwasniewski, M. Naumann, T. Schulz, R. Schewski, and D. Klimm, J. Cryst. Growth 404, 184 (2014).

10M. Handwerg, R. Mitdank, Z. Galazka, and S. Fischer, Semicond. Sci. Technol. 30, 024006 (2015).

11P. Jiang, X. Qian, X. Li, and R. Yang, Appl. Phys. Lett. 113, 232105 (2018).

12Z. Liu and T. Luo, in Gallium Oxide: Materials Properties, Crystal Growth, and Devices, edited by M. Higashiwaki and S. Fujita (Springer International Publishing, Cham, 2020), p. 535.

13M. D. Santia, N. Tandon, and J. Albrecht, Appl. Phys. Lett. 107, 041907 (2015).

14Z. Yan and S. Kumar, Phys. Chem. Chem. Phys. 20, 29236 (2018).

15C. Carbogno, R. Ramprasad, and M. Scheffler, Phys. Rev. Lett. 118, 175901 (2017).

16K. Kadau, T. C. Germann, and P. S. Lomdahl, Int. J. Mod. Phys. C 17, 1755 (2006).

17R. A. B€ockmann and H. Grubmuller, € Nat. Struct. Biol. 9, 198 (2002).

18J. R. Lloyd and T. Luo, Handbook of Molecular Dynamics Potential Functions (Begell House, Inc., Redding, CT, 2011).

19X. Wan, W. Feng, Y. Wang, H. Wang, X. Zhang, C. Deng, and N. Yang, Nano Lett. 19, 3387 (2019).

20J. Behler and M. Parrinello, Phys. Rev. Lett. 98, 146401 (2007).

21A. P. Bart-ok, M. C. Payne, R. Kondor, and G. Cs-anyi, Phys. Rev. Lett. 104, 136403 (2010).

22A. P. Bart-ok, J. Kermode, N. Bernstein, and G. Cs-anyi, Phys. Rev. X 8, 041048 (2018).

23X. Qian and R. Yang, Phys. Rev. B 98, 224108 (2018).

24P. Rowe, G. Cs-anyi, D. Alfe\`, and A. Michaelides, Phys. Rev. B 97, 054303 (2018).

25X. Gu and C. Zhao, Comput. Mater. Sci. 165, 74 (2019).

26G. C. Sosso, V. L. Deringer, S. R. Elliott, and G. Cs-anyi, Mol. Simul. 44, 866 (2018).

27R. Galvelis, S. Doerr, J. M. Damas, M. J. Harvey, and G. De Fabritiis, J. Chem. Inf. Model. 59, 3485 (2019).

28R. Li, E. Lee, and T. Luo, Mater. Today Phys. 12, 100181 (2020).

29A. Rohskopf, S. Wyant, K. Gordiz, H. R. Seyf, M. G. Muraleedharan, and A. Henry, Comput. Mater. Sci. 184, 109884 (2020).

30R. Car and M. Parrinello, Phys. Rev. Lett. 55, 2471 (1985).

31J. VandeVondele, M. Krack, F. Mohamed, M. Parrinello, T. Chassaing, and J. Hutter, Comput. Phys. Commun. 167, 103 (2005).

32T. D. Kuhne, M. Krack, F. R. Mohamed, and M. Parrinello, € Phys. Rev. Lett. 98, 066401 (2007).

33S. Goedecker, M. Teter, and J. Hutter, Phys. Rev. B 54, 1703 (1996).

34J. A˚ hman, G. Svensson, and J. Albertsson, Acta Crystallogr., Sect. C 52, 1336 (1996).

35L. Zhang, J. Han, H. Wang, W. Saidi, R. Car, and E. Weinan, in Advances in Neural Information Processing Systems (NIPS, 2018), p. 4436–4446.

36L. Zhang, J. Han, H. Wang, R. Car, and E. Weinan, Phys. Rev. Lett. 120, 143001 (2018).

37H. Wang, L. Zhang, J. Han, and E. Weinan, Comput. Phys. Commun. 228, 178 (2018).

38D. P. Kingma and J. Ba, arXiv:1412.6980 (2014).

39See https://github.com/RuiyangLi6/NNP_Ga2O3 for data and code.

40S. Plimpton, J. Comput. Phys. 117(1), 1 (1995).

41J. Tersoff, Phys. Rev. B 39, 5566 (1989).

42F. H. Stillinger and T. A. Weber, Phys. Rev. B 31, 5262 (1985).

43A. Rohskopf, H. R. Seyf, K. Gordiz, T. Tadano, and A. Henry, npj Comput. Mater. 3(1), 27 (2017).

44T. Tadano, Y. Gohda, and S. Tsuneyuki, J. Phys. 26, 225402 (2014).

45M. A. Blanco, M. B. Sahariah, H. Jiang, A. Costales, and R. Pandey, Phys. Rev. B 72, 184103 (2005).

46D. Ma, G. Zhang, and L. Zhang, J. Phys. D 53, 434001 (2020).

47X. Wu, J. Lee, V. Varshney, J. L. Wohlwend, A. K. Roy, and T. Luo, Sci. Rep. 6, 22504 (2016).

48R. Kubo, M. Toda, and N. Hashitsume, Statistical Physics II: Nonequilibrium Statistical Mechanics (Springer Science & Business Media, 2012), Vol. 31.

49H. Meng, X. Yu, H. Feng, Z. Xue, and N. Yang, Int. J. Heat Mass Transfer 137, 1241 (2019).

50X. Yu, R. Li, T. Shiga, L. Feng, M. An, L. Zhang, J. Shiomi, and N. Yang, J. Phys. Chem. C 123, 26735 (2019).

51Z. Fan, L. F. C. Pereira, H.-Q. Wang, J.-C. Zheng, D. Donadio, and A. Harju, Phys. Rev. B 92, 094301 (2015).

52A. Kinaci, J. B. Haskins, and T. C¸agın, J. Chem. Phys. 137, 014106 (2012).

53T. Feng, L. Lindsay, and X. Ruan, Phys. Rev. B 96, 161201 (2017).

54J. Turney, A. McGaughey, and C. Amon, Phys. Rev. B 79, 224305 (2009).

55W. Lv and A. Henry, New J. Phys. 18, 013028 (2016).

56H. R. Seyf, K. Gordiz, F. DeAngelis, and A. Henry, J. Appl. Phys. 125, 081101 (2019).

---
