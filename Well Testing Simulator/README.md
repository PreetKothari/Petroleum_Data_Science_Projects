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

The shown parameters and [WTA.xlsx](https://github.com/PreetKothari/Petroleum_Data_Science_Projects/blob/main/Well%20Testing%20Simulator/WTA.xlsx) dataset was taken for modelling well behavior. 

![image](https://user-images.githubusercontent.com/87279526/235211045-1901442c-d213-42df-8175-55797f6d846d.png)

### Well Test Plots

The following plots were plotted:

![image](https://user-images.githubusercontent.com/87279526/235212561-0252dac5-233d-494b-98d5-d7c4d328ed13.png)
![image](https://user-images.githubusercontent.com/87279526/235212581-7992bb35-2971-429e-b5db-8f9e721c3682.png)
![image](https://user-images.githubusercontent.com/87279526/235212595-60814be3-0054-473b-a656-380f3922d600.png)
![image](https://user-images.githubusercontent.com/87279526/235212612-6ffb76b4-67c3-4911-9b53-59ab2d0cb7ae.png)
![image](https://user-images.githubusercontent.com/87279526/235212631-137c1d26-3a80-4d95-a99e-94ed4da86795.png)
![image](https://user-images.githubusercontent.com/87279526/235212653-46706038-b091-465c-aa4f-6a9b89664f3c.png)


Using the Semi-Log and Log-Log plots the Fully Wellbore Storage (FWBS) and the Infinite Acting Radial Flow (IARF) periods were estimated.

![image](https://user-images.githubusercontent.com/87279526/235213272-1fc3f511-aaf3-4774-b2e4-3b61480827e9.png)

![image](https://user-images.githubusercontent.com/87279526/235213209-85a4b5d7-7efe-40b7-a0cf-dabdf6ae4843.png)

FWBS period exists around t = 0.001 to t = 0.04 hours.

![image](https://user-images.githubusercontent.com/87279526/235213301-784a9761-5f7b-461c-b482-e8b87b2852ee.png)

![image](https://user-images.githubusercontent.com/87279526/235213324-8e0f16ab-4f27-440a-8d20-a6d984cc8e6c.png)

IARF period exists around t = 10 to t = 65 hours.

Now, after ascertaining the FWBS and IARF periods, for further analysis and calculation of well and reservoir parameters like Wellbore Storage and Skin, Permeability, etc. two approaches were used, namely:

1. Conventional Method –

From the log-log plot it was estimated that the FWBS period exists around t = 0 to t = 0.04 hours.

![image](https://user-images.githubusercontent.com/87279526/235213644-40029395-cb3b-4468-89d6-f02d421a7791.png)

The different values of Pwf were plotted against time for the fully wellbore storage period on a Cartesian plot to calculate the slope of the graph and find the wellbore storage constant. 

![image](https://user-images.githubusercontent.com/87279526/235213917-e12177e3-9273-4843-a084-cbcdd2d956a1.png)

