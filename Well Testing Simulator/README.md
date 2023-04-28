# Well Test Simulator

Objective is to develop a Well Testing Simulator using Python for interpretation of a well test using Log-Log and Semi-Log plots to identify various flow regimes and calculation of Skin, Permeability, and other such parameters.for rate of penetration (ROP) during oil & gas drilling operations.

### Well Test Analysis

Well-test is the activity of opening a well to flow fluids from and into the reservoir (or production and injection, respectively) and shutting-in for several hours or days. The objective is not to produce the fluid to be considered as the economic value but to acquire data for understanding the reservoir.

During a well test, the response of a reservoir to changing production or injection conditions is monitored. Since the response is, to a greater or lesser degree, characteristic of the properties of the reservoir it is possible in many cases to infer reservoir properties from the response. Well test interpretation is therefore an inverse problem in that the model parameters are inferred by analysing the model response to a given input.

### Single-Phase Fluid Flow in Porous Media 

The simulation is based upon the Radial Darcy's flow, which is a flow of fluid directing radially from a cylindrical reservoir into the wellbore, or vice versa.

Assumptions for Darcy's flow are: 
  * Single-phase flow
  * Laminar flow
  * Homogeneous reservoir properties
  * Slightly compressible fluid

The last two assumptions are approximately applicable to oil and water under an isothermal condition, and gases under high-pressure. The flow is formulated as a partial differential equation (PDE) such that the pressure p derivative of the radial distance r to the wellbore and the time t, this equation is known as the Diffusivity Equation.

                 (𝜹^𝟐 𝒑)/(𝜹𝒓^𝟐) + (𝟏/𝒓)x(𝜹𝒑/𝜹𝒓) = ((𝝓𝒄_𝒕 𝝁)/𝒌)x(𝜹𝒑/𝜹𝒕)

### Constant Terminal Rate Solution for Transient Flow Conditions

When the flow is not yet reaching the outer reservoir boundary, the flow is called to behave infinite-acting and during this time the reservoir is in Transient State condition. Assuming Transient flow conditions, the solution to the radial diffusivity equation is based on initial and boundary conditions: 
* Initial Condition - Reservoir pressure is at its initial uniform value. 
                 
                 𝑃(𝑟,𝑡) = 𝑃𝑖, 𝑎𝑡 𝑡 = 0 𝑓𝑜𝑟 𝑎𝑙𝑙 𝑟
* Outer Boundary Condition - The pressure remains unaffected at the reservoir boundary during the infinite-acting flow period. 
                 
                 𝜕𝑃/𝜕𝑡 = 0, 𝑎𝑡 𝑟 = ∞ and 𝑃 = 𝑃𝑖, 𝑎𝑡 𝑟 = ∞ 𝑓𝑜𝑟 𝑎𝑙𝑙 𝑡
* Inner Boundary Condition - A constant flow rate at the wellbore is assumed.
                 
                 𝑞 = (𝑘(2𝜋𝑟ℎ)/𝜇)x(𝜕𝑃/𝜕𝑟)𝑟=𝑟𝑤 , 𝑎𝑡 𝑟 = 𝑟𝑤 and lim𝑟→0 𝑟x(𝜕𝑃/𝜕𝑟) = 𝑞𝜇/2𝜋𝑘h

After solving the diffusivity equation based on above conditions:

![image](https://user-images.githubusercontent.com/87279526/235210374-3456e75d-a98c-4164-8655-b97cbd3930cd.png)

### Modelling in Python

The shown parameters and dataset was taken for modelling well behavior. 

![image](https://user-images.githubusercontent.com/87279526/235211045-1901442c-d213-42df-8175-55797f6d846d.png)




Despite many efforts (theoretical and experimental) throughout the years, modelling the ROP as a mathematical function of some key variables is not so trivial, due to the highly non-linearity behaviour experienced. Therefore, several researches in the recent years have been proposing to use data-driven models from artificial intelligence field for ROP prediction and optimization.

Our aim is to encourage participants from both domain and non-domain to come together and find innovative solutions to address the challenge.

### Datasets Description
The dataset is derived from Equinor’s public Volve dataset under Equinor Open Data Licence. The dataset contains seven wells with twelve common drilling attributes with nearly
200,000 samples.
