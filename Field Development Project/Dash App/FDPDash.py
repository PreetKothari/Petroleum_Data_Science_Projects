#!/usr/bin/env python
# coding: utf-8

# ## **Modelling Field Development in O&G**

# Objectives: 
# * Model Waterflood Recovery and Recovery after EOR based on given Swi, Sor,
#   muo, and muw. 
# * Comparision between EOR & Waterflood Recovery.
# * Generating Relative Permeablity and Fractional Flow Curves.
# * Generating Production Profiles over a time period for given Area, OIIP, 
#   Test Liquid Rate and Well Spacing pattern.
# * Basic Economic Analysis based on different Production Profiles.
# * Generating Petroleum Project Net Cash Flows (NCF), Net Present Value (NPV),
#   Internal Rate of Return (IRR) and Different plots for in-depth Economic 
#   Analysis based on given CAPEX, OPEX, Tax, Royalty, and Oil Prices.
# * Sensitivity Analysis through Tornado Charts and Spider Plots.
# 
# @author: Preet Kothari
# 
# @email: preetkothari19@gmail.com
# 
# Note: 
# Depreciation is still to be included.

# ### **Functions**

# #### Recovery Analysis Functions - Waterflood & EOR

# In[ ]:


def Rec_Analysis(Swi=0,Sorw=0,muo=0,muw=0):
    """Collects values of Water Saturation (Swi), Residual Oil Saturation (Sor),
     Oil Viscosity (muo), and Water Viscosity (muw). 
     Calculates So, Kro, Krw, M, and fw.
     Returns a Dataframe with R, Sw, So, Sw*, Kro, Krw, M, fw"""
    import pandas as pd
    import numpy as np

    dict_data = {'Swi':0,'Sorw':0,'muo':0,'muw':0}
    if (Swi==0):
        dict_data['Swi'] = float(input("Enter Initial Water Saturation, Swi: "))
        dict_data['Soi'] = 1-dict_data['Swi']
        print('Initial Oil Saturation, Soi:',dict_data['Soi'])
        dict_data['Sorw'] = float(input("Enter Residual Oil Saturation to Water, Sorw: "))
        dict_data['muo'] = float(input("Enter Oil Viscosity, Muo: "))
        dict_data['muw'] = float(input("Enter Water Viscosity, Muw: "))
    else:
        dict_data['Swi'] = Swi
#         print('Initial Water Saturation, Swi:',dict_data['Swi'])
        dict_data['Soi'] = 1-Swi
#         print('Initial Oil Saturation, Soi:',dict_data['Soi'])
        dict_data['Sorw'] = Sorw
#         print('Residual Oil Saturation to Water, Sorw:',dict_data['Sorw'])
        dict_data['muo'] = muo
#         print('Oil Viscosity, Muo:',dict_data['muo'])
        dict_data['muw'] = muw
#         print('Water Viscosity, Muw:',dict_data['muw'])

    R = np.arange(0,0.61,0.01)                                               # Recovery
    Sw = R*(1-dict_data['Swi']) + dict_data['Swi']                           # Water Saturation
    So = (1-Sw)                                                              # Oil Saturation
    Sw_norm = (Sw-dict_data['Swi'])/(1-dict_data['Swi']-dict_data['Sorw'])   # Normalized Water Saturation
    Krw = (Sw_norm)**3                                                       # Relative Permeabiltiy to Water
    Kro = (1-Sw_norm)**3                                                     # Relative Permeabiltiy to Oil
    M = (Krw*dict_data['muo'])/(Kro*dict_data['muw'])                        # Mobility Contrast
    Fw = 1/(1+(1/M))                                                         # Fractional Water Cut

    data = pd.DataFrame(data={'Recovery':R,'Oil Saturation':So,'Water Saturation':Sw,'Normalized Water Saturation':Sw_norm,'Water Rel Perm':Krw,'Oil Rel Perm':Kro,'Mobility Contrast':M,'Water Cut':Fw})
    return data


# In[ ]:


def Rel_Perm_Curve(data):
    """Plots Kro & Krw vs Sw"""
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    # Relative Permeability Curve
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['Water Saturation'],
                              y=data['Oil Rel Perm'],
                              name='Kro',
                              line=dict(color='red', width=4)))
    fig.add_trace(go.Scatter(x=data['Water Saturation'],
                              y=data['Water Rel Perm'],
                              name='Krw',
                              line=dict(color='blue', width=4)))
    fig.add_trace(go.Scatter(x=np.full((data.shape[0]),0.5),
                              y=data['Oil Rel Perm'],
                              line=dict(color='black', width=2, dash='dash'),
                             showlegend=False))
    fig.update_layout(title='<b>Relative Permeability Curve</b>',
                      title_x=0.5,
                      xaxis_title='Water Saturation, Sw',
                      xaxis_range=[0,1],
                      yaxis_title='Rel Perm, Krw & Kro',
                      yaxis_range=[0,1])
    return fig


# In[ ]:


def Curves(data,state):
    """Plots Rec vs Sw & fw vs Sw"""
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Fractional Flow Curve','Recovery vs Water Cut'))

    # Fractional Flow Curve
    fig.add_trace(go.Scatter(x=data['Water Saturation'],
                             y=data['Water Cut'],
                             mode='lines+markers',
                             line=dict(color='purple', width=4),
                             showlegend=False),
                  row=1, col=1)

    # Recovery vs Water Cut
    fig.add_trace(go.Scatter(x=data['Recovery'],
                             y=data['Water Cut'],
                             mode='lines+markers',
                             line=dict(color='limegreen', width=4),
                             showlegend=False),
                  row=1, col=2)
    
    # Update x axis properties
    fig.update_xaxes(title_text="Water Saturation, Sw", range=[0,0.8], row=1, col=1)
    fig.update_xaxes(title_text="Recovery, R", range=[0,0.8], row=1, col=2)
    
    # Update y axis properties
    fig.update_yaxes(title_text="Water Cut, Fw", range=[0,1], row=1, col=1)
    fig.update_yaxes(title_text="Water Cut, Fw", range=[0,1], row=1, col=2)
    
    fig.update_layout(title='<b>Fractional Flow and Recovery - '+state+'</b>',
                      title_x=0.5)

    return fig


# In[ ]:


def EOR_Curves(data_bf_eor,data_af_eor):
    """Plots Rec vs Sw & fw vs Sw for both before and after EOR"""
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    r_coef_bf_eor,r_intercept_bf_eor = Poly_eq(6,data_bf_eor.iloc[:,0:1].values,data_bf_eor.iloc[:,-1:].values)
    r_coef_af_eor,r_intercept_af_eor = Poly_eq(6,data_af_eor.iloc[:,0:1].values,data_af_eor.iloc[:,-1:].values)
    sw_coef_bf_eor,sw_intercept_bf_eor = Poly_eq(6,data_bf_eor.iloc[:,2:3].values,data_bf_eor.iloc[:,-1:].values)
    sw_coef_af_eor,sw_intercept_af_eor = Poly_eq(6,data_af_eor.iloc[:,2:3].values,data_af_eor.iloc[:,-1:].values)

    fwr_bf = round(Sol(r_coef_bf_eor,r_intercept_bf_eor),3)
    fwr_af = round(Sol(r_coef_af_eor,r_intercept_af_eor),3)
    fws_bf = round(Sol(sw_coef_bf_eor,sw_intercept_bf_eor),3)
    fws_af = round(Sol(sw_coef_af_eor,sw_intercept_af_eor),3)
    
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Fractional Flow Curve','Recovery vs Water Cut'))

    # Fractional Flow Curve - Before and After EOR
    fig.add_trace(go.Scatter(x=data_bf_eor['Water Saturation'],
                             y=data_bf_eor['Water Cut'],
                             mode='lines+markers',
                             name='Before EOR',
                             line=dict(color='purple', width=4),
                             showlegend=True),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data_af_eor['Water Saturation'],
                             y=data_af_eor['Water Cut'],
                             mode='lines+markers',
                             name='After EOR',
                             line=dict(color='deepskyblue', width=4),
                             showlegend=True),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(0,(float(fws_af)+0.01),float(fws_af)),
                             y=np.full(2,0.9),
                             line=dict(color='black', width=2, dash='dash'),
                             showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=np.full(2,float(fws_bf)),
                             y=np.arange(0,0.91,0.9),
                             line=dict(color='black', width=2, dash='dash'),
                             showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=np.full(2,float(fws_af)),
                             y=np.arange(0,0.91,0.9),
                             line=dict(color='black', width=2, dash='dash'),
                             showlegend=False),
                  row=1, col=1)

    # Recovery vs Water Cut - Before and After EOR
    fig.add_trace(go.Scatter(x=data_bf_eor['Recovery'],
                             y=data_bf_eor['Water Cut'],
                             mode='lines+markers',
                             name='Before EOR',
                             line=dict(color='limegreen', width=4),
                             showlegend=True),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=data_af_eor['Recovery'],
                             y=data_af_eor['Water Cut'],
                             mode='lines+markers',
                             name='After EOR',
                             line=dict(color='red', width=4),
                             showlegend=True),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=np.arange(0,(float(fwr_af)+0.01),float(fwr_af)),
                             y=np.full(2,0.9),
                             line=dict(color='black', width=2, dash='dash'),
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=np.full(2,float(fwr_bf)),
                             y=np.arange(0,0.91,0.9),
                             line=dict(color='black', width=2, dash='dash'),
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=np.full(2,float(fwr_af)),
                             y=np.arange(0,0.91,0.9),
                             line=dict(color='black', width=2, dash='dash'),
                             showlegend=False),
                  row=1, col=2)

    # Update x axis properties
    fig.update_xaxes(title_text="Water Saturation, Sw", range=[0,0.8], row=1, col=1)
    fig.update_xaxes(title_text="Recovery, R", range=[0,0.8], row=1, col=2)
    
    # Update y axis properties
    fig.update_yaxes(title_text="Water Cut, Fw", range=[0,1], row=1, col=1)
    fig.update_yaxes(title_text="Water Cut, Fw", range=[0,1], row=1, col=2)

    fig.update_layout(title='<b>Comparision - WF vs EOR</b>',
                      title_x=0.5)

    text='The approximate recovery at 90% water cut before EOR was '+str(int(fwr_bf*100))+'% and after applying EOR the approximate recovery was '+str(int(fwr_af*100))+'%. The approximate water saturation at 90% water cut before EOR was '+str(int(fws_bf*100))+'% and after applying EOR the approximate water saturation was '+str(int(fws_af*100))+'%.'

    return fig,text


# #### Functions for Generating 6 Degree Polynomial Equations and their Solutions

# In[ ]:


def Poly_eq(degree,x,y):
    """Generates a degree 6 polynomial equation for R and fw"""
    import pandas as pd
    import numpy as np

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    polynomial_features= PolynomialFeatures(degree=6)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    return model.coef_, model.intercept_

def Sol(coef,intercept):
    """Solves a degree 6 polynomial equation for R and fw for fw = 90%"""
    import pandas as pd
    import numpy as np

    from sympy.solvers import solve
    from sympy import symbols

    x = symbols('x')
    val = solve((coef[0][6]*(x**6)) + (coef[0][5]*(x**5)) + (coef[0][4]*(x**4)) + (coef[0][3]*(x**3)) + (coef[0][2]*(x**2)) + (coef[0][1]*x) + (intercept[0]) - 0.9, x)
    return val[1]

def Fractional_water_cut(x,coef,intercept):
    return (coef[0][6]*(x**6)) + (coef[0][5]*(x**5)) + (coef[0][4]*(x**4)) + (coef[0][3]*(x**3)) + (coef[0][2]*(x**2)) + (coef[0][1]*x) + (intercept)


# #### Production Profile Functions

# In[ ]:


def Prod_dictionary(A=0,Oiip=0,Years=25,test_liq_rate=0):
    """Collects values of Area (A), Oil Initially In Place (OIIP), Years of Project (Years),
     and Test LIquid Rate (test_liq_rate)"""
    import pandas as pd
    import numpy as np

    dict_prod = {'Area':0,'OIIP':0,'Years':0,'Test Liq. Rate':0}
    if (A==0):
        dict_prod['Area'] = float(input("Enter Area, Km2: "))
        dict_prod['OIIP'] = float(input("Enter Oil Initially In PLace, MMm3: "))
        dict_prod['Years'] = float(input("Enter Years for Production: "))
        dict_prod['Test Liq. Rate'] = float(input("Enter Test Liq. Rate, m3/d: ")) 
    else:
        dict_prod['Area'] = A                       # Area in Km^2
        dict_prod['OIIP'] = Oiip                    # Oil Initially in PLace in MMm^3
        dict_prod['Years'] = Years                  # Active years
        dict_prod['Test Liq. Rate'] = test_liq_rate # Test liquid rate per well in m^3/d

    return dict_prod


# In[ ]:


def Well_req(dict_prod,spacing):
    """Calculates number of wells required based on well spacing pattern"""
    import pandas as pd
    import numpy as np

    # To ignore warnings generated by the current version of seaborn 
    import warnings                  
    warnings.filterwarnings("ignore")

    spacing = float(spacing)/1000
    return int((dict_prod['Area']*0.9)/(spacing**2))


# In[ ]:


def Drilled_wells(w,dict_prod):
    """Calculates maximum, total drilled and producing wells in each year based on 
     number of wells required & well spacing pattern"""
    import pandas as pd
    import numpy as np

    # To ignore warnings generated by the current version of seaborn 
    import warnings                  
    warnings.filterwarnings("ignore")

    max_wells = [0]*dict_prod['Years']
    total_wells_drilled = [0]*dict_prod['Years']
    prod_wells = [0]*dict_prod['Years']
    for i in range(dict_prod['Years']):
        if (i==0):
            max_wells[i] = 24 if w>=24 else w
            total_wells_drilled[i] = 24 if w>=24 else w
            we = w
            f = np.zeros(12)
            ind = 2
            while (we>1) and (ind<12):
                f[ind]=2*(12-ind)/12
                we = we-2
                ind = ind+1
            if (we==1) and (ind<12) :
                f[ind]=1*(12-ind)/12
            prod_wells[i] = sum(f)  
        else:
            a = (total_wells_drilled[i-1]-4)+2+(11/6)
            if (total_wells_drilled[i-1]==w):
                max_wells[i]=0
            elif (w>(total_wells_drilled[i-1]+24)):
                max_wells[i]=24 
            else:
                max_wells[i]=(w-total_wells_drilled[i-1])

            total_wells_drilled[i] = total_wells_drilled[i-1]+max_wells[i]
            if (max_wells[i]==24):
                prod_wells[i]=a+prod_wells[0]
            elif ((max_wells[i]==0) and ((a+prod_wells[0])>total_wells_drilled[i])):
                prod_wells[i]=a+total_wells_drilled[i]-a
            else:
                prod_wells[i]=a+((max_wells[i]/2)*(20-((max_wells[i]/2)-1))/12)

    return np.array(max_wells), np.array(total_wells_drilled), np.array(prod_wells)


# In[ ]:


def Production_profile(dict_prod,coef,intercept,spacing=0):
    """Returns a dataframe with average oil and water production rates 
     based on field & reservoir parameters and well spacing pattern"""
    import pandas as pd
    import numpy as np

    # To ignore warnings generated by the current version of seaborn 
    import warnings                  
    warnings.filterwarnings("ignore")  

    if (spacing==0):
        spacing = float(input("Enter Well Spacing for generating Production Profile: "))
    n_wells_req = Well_req(dict_prod,spacing)
    max_wells, total_wells_drilled, prod_wells = Drilled_wells(n_wells_req,dict_prod)

    yr = np.zeros(dict_prod['Years'])
    liq_rate = prod_wells*dict_prod['Test Liq. Rate']
    apx_water_cut = np.zeros(dict_prod['Years'])
    oil_rate = np.zeros(dict_prod['Years'])
    apx_cum_oil = np.zeros(dict_prod['Years'])
    rec = np.zeros(dict_prod['Years'])
    end_wc = np.zeros(dict_prod['Years'])
    avg_water = np.zeros(dict_prod['Years'])
    avg_oil_rate = np.zeros(dict_prod['Years'])
    cum_oil = np.zeros(dict_prod['Years'])
  
    for i in range(dict_prod['Years']):
        yr[i] = i+1
        if (i==0):
            oil_rate[i] = liq_rate[i]*(1-apx_water_cut[i])
            apx_cum_oil[i] = (oil_rate[i]*365)/1000000
            rec[i] = (apx_cum_oil[i]/dict_prod['OIIP']) 
            if (Fractional_water_cut(rec[i],coef,intercept)<0):
                end_wc[i] = 0
            else:
                end_wc[i] = Fractional_water_cut(rec[i],coef,intercept)
            avg_water[i] = (end_wc[i]+apx_water_cut[i])/2
            avg_oil_rate[i] = liq_rate[i]*(1-avg_water[i])
            cum_oil[i] = (avg_oil_rate[i]*365)/1000000
        else:
            apx_water_cut[i] = end_wc[i-1]
            oil_rate[i] = liq_rate[i]*(1-apx_water_cut[i])
            apx_cum_oil[i] = ((oil_rate[i]*365)/1000000)+apx_cum_oil[i-1]
            rec[i] = (apx_cum_oil[i]/dict_prod['OIIP'])
            if (Fractional_water_cut(rec[i],coef,intercept)<0):
                end_wc[i] = 0
            else:
                end_wc[i] = Fractional_water_cut(rec[i],coef,intercept) 
            avg_water[i] = (end_wc[i]+apx_water_cut[i])/2
            avg_oil_rate[i] = liq_rate[i]*(1-avg_water[i])
            cum_oil[i] = ((avg_oil_rate[i]*365)/1000000)+cum_oil[i-1]

    oil_rec = (100*cum_oil)/dict_prod['OIIP']
    avg_water_rate = liq_rate*avg_water
    data = pd.DataFrame(data={'Year':yr,'Max Wells in Year':max_wells,'Flowing Wells':prod_wells,
                            'Avg. Water-Cut, %':avg_water,'Avg. Oil Rate, m3/d':avg_oil_rate,
                            'Avg. Water Rate, m3/d':avg_water_rate,'Cum. Oil, MMm3':cum_oil,'Recovery, %':oil_rec})
    # data = pd.DataFrame(data={'Year':yr,'Max Wells in Year':max_wells,'Total Wells Drilled':total_wells_drilled,'Flowing Wells':prod_wells,
    #                           'Liq. rate':liq_rate,'Approx Water-cut':apx_water_cut,'Oil rate':oil_rate,'Approx Cum Oil':apx_cum_oil,
    #                           'Np/N':rec,'End of year wc':end_wc,'Avg. Water-Cut, %':avg_water,'Avg. Oil Rate, m3/d':avg_oil_rate,
    #                           'Cum. Oil, MMm3':cum_oil,'Recovery, %':oil_rec,'Avg. Water rate':avg_water_rate})
    return data


# In[ ]:


def Avg_rate_stacked_bar(profile,state):
    """Stacked Barplot of Average Oil and Water and Flow Rates"""
    
    import plotly.express as px
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[
        go.Bar(x=profile['Year'],
               y=profile['Avg. Oil Rate, m3/d'],
               name='Avg. Oil Rate',
               marker_color='orangered'),
        go.Bar(x=profile['Year'],
               y=profile['Avg. Water Rate, m3/d'],
               name='Avg. Water Rate',
               marker_color='blue')
    ])
    
    fig.update_layout(title='<b>Year-wise Average Oil and Water Flow Rate - '+state+'</b>',
                      title_x=0.5,
                      xaxis_title='Years',
                      yaxis_title='Flow Rate, m3/d',
                      barmode='stack')
    return fig


# #### Economical Analysis Functions

# In[ ]:


def Eco_curves(dict_prod,coef,intercept,state):
    """Smooth Scatter Plot of Yearly Average Oil Rate, Yearly Recovery, and Water 
     Yearly Average Water Cut for different well spacing patterns (700 m, 600 m,..., 200 m)"""
    
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    profile_7 = Production_profile(dict_prod,coef,intercept,700)
    profile_6 = Production_profile(dict_prod,coef,intercept,600)
    profile_5 = Production_profile(dict_prod,coef,intercept,500)
    profile_4 = Production_profile(dict_prod,coef,intercept,400)
    profile_3 = Production_profile(dict_prod,coef,intercept,300)
    profile_2 = Production_profile(dict_prod,coef,intercept,200)

    fig = make_subplots(rows=1, cols=3,
                       subplot_titles=('Yearly Oil Rate based <br> on Well Spacing',
                                       'Yearly Recovery based <br> on Well Spacing',
                                       'Yearly Avg. Water Cut based <br> on Well Spacing'))

    # Oil Rate vs Year
    fig.add_trace(go.Scatter(x=profile_7['Year'],
                             y=profile_7['Avg. Oil Rate, m3/d'],
                             name='700 m',
                             line=dict(color='indigo', width=4),
                             legendgroup='7',
                             showlegend=True),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=profile_6['Year'],
                             y=profile_6['Avg. Oil Rate, m3/d'],
                             name='600 m',
                             line=dict(color='deepskyblue', width=4),
                             legendgroup='6',
                             showlegend=True),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=profile_5['Year'],
                             y=profile_5['Avg. Oil Rate, m3/d'],
                             name='500 m',
                             line=dict(color='limegreen', width=4),
                             legendgroup='5',
                             showlegend=True),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=profile_4['Year'],
                             y=profile_4['Avg. Oil Rate, m3/d'],
                             name='400 m',
                             line=dict(color='yellow', width=4),
                             legendgroup='4',
                             showlegend=True),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=profile_3['Year'],
                             y=profile_3['Avg. Oil Rate, m3/d'],
                             name='300 m',
                             line=dict(color='orange', width=4),
                             legendgroup='3',
                             showlegend=True),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=profile_2['Year'],
                             y=profile_2['Avg. Oil Rate, m3/d'],
                             name='200 m',
                             line=dict(color='red', width=4),
                             legendgroup='2',
                             showlegend=True),
                  row=1, col=1)

    # Recovery vs Year
    fig.add_trace(go.Scatter(x=profile_7['Year'],
                             y=profile_7['Recovery, %'],
                             name='700 m',
                             line=dict(color='indigo', width=4),
                             legendgroup='7',
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=profile_6['Year'],
                             y=profile_6['Recovery, %'],
                             name='600 m',
                             line=dict(color='deepskyblue', width=4),
                             legendgroup='6',
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=profile_5['Year'],
                             y=profile_5['Recovery, %'],
                             name='500 m',
                             line=dict(color='limegreen', width=4),
                             legendgroup='5',
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=profile_4['Year'],
                             y=profile_4['Recovery, %'],
                             name='400 m',
                             line=dict(color='yellow', width=4),
                             legendgroup='4',
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=profile_3['Year'],
                             y=profile_3['Recovery, %'],
                             name='300 m',
                             line=dict(color='orange', width=4),
                             legendgroup='3',
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=profile_2['Year'],
                             y=profile_2['Recovery, %'],
                             name='200 m',
                             line=dict(color='red', width=4),
                             legendgroup='2',
                             showlegend=False),
                  row=1, col=2)
    
    # Avg. Water Cut vs Year
    fig.add_trace(go.Scatter(x=profile_7['Year'],
                             y=profile_7['Avg. Water-Cut, %'],
                             name='700 m',
                             line=dict(color='indigo', width=4),
                             legendgroup='7',
                             showlegend=False),
                  row=1, col=3)
    fig.add_trace(go.Scatter(x=profile_6['Year'],
                             y=profile_6['Avg. Water-Cut, %'],
                             name='600 m',
                             line=dict(color='deepskyblue', width=4),
                             legendgroup='6',
                             showlegend=False),
                  row=1, col=3)
    fig.add_trace(go.Scatter(x=profile_5['Year'],
                             y=profile_5['Avg. Water-Cut, %'],
                             name='500 m',
                             line=dict(color='limegreen', width=4),
                             legendgroup='5',
                             showlegend=False),
                  row=1, col=3)
    fig.add_trace(go.Scatter(x=profile_4['Year'],
                             y=profile_4['Avg. Water-Cut, %'],
                             name='400 m',
                             line=dict(color='yellow', width=4),
                             legendgroup='4',
                             showlegend=False),
                  row=1, col=3)
    fig.add_trace(go.Scatter(x=profile_3['Year'],
                             y=profile_3['Avg. Water-Cut, %'],
                             name='300 m',
                             line=dict(color='orange', width=4),
                             legendgroup='3',
                             showlegend=False),
                  row=1, col=3)
    fig.add_trace(go.Scatter(x=profile_2['Year'],
                             y=profile_2['Avg. Water-Cut, %'],
                             name='200 m',
                             line=dict(color='red', width=4),
                             legendgroup='2',
                             showlegend=False),
                  row=1, col=3)
    
    # Update x axis properties
    fig.update_xaxes(title_text="Year", range=[0,25], row=1, col=1)
    fig.update_xaxes(title_text="Year", range=[0,25], row=1, col=2)
    fig.update_xaxes(title_text="Year", range=[0,25], row=1, col=3)
    
    # Update y axis properties
    fig.update_yaxes(title_text="Oil Rate, m3/d", rangemode="tozero", title_standoff = 0, row=1, col=1)
    fig.update_yaxes(title_text="Recovery, %", rangemode="tozero", title_standoff = 0, row=1, col=2)
    fig.update_yaxes(title_text="Avg. Water Cut, %", rangemode="tozero", title_standoff = 0, row=1, col=3)
    
    
    fig.update_layout(title_text="<b>"+state+"</b>", title_x=0.5)

    return fig


# In[ ]:


def Exp_dictionary(capex=0,opex=0,well_cost=0,oil_price=0,dis_rate=0,royalty_rate=0,tax_rate=0):
    """Collects values of CAPEX, OPEX, Well Cost, Oil Price, Discount Rate,
     Royalty Rate and Tax Rate"""
    import pandas as pd
    import numpy as np

    # To ignore warnings generated by the current version of seaborn 
    import warnings                  
    warnings.filterwarnings("ignore")   

    dict_exp = {'Capex':0,'Opex':0,'Well Cost':0,'Oil Price':0,'Discount Rate':0,'Royalty Rate':0,'Tax Rate':0}
    if (capex==0):
        dict_exp['Capex'] = float(input("Enter Capital Expenses, Mn $: ")) #capex
        dict_exp['Opex'] = float(input("Enter Operational Expenses, $/bbl: ")) #opex
        dict_exp['Well Cost'] = float(input("Enter Well Cost, Mn $/well: ")) #well_cost
        dict_exp['Oil Price'] = float(input("Enter Oil Price, $/bbl: ")) #oil_price
        dict_exp['Discount Rate'] = float(input("Enter Discount Rate: ")) #dis_rate
        dict_exp['Royalty Rate'] = float(input("Enter Royalty Rate: ")) #royalty_rate
        dict_exp['Tax Rate'] = float(input("Enter Tax Rate: ")) #tax_rate
    else:
        dict_exp['Capex'] = capex
        dict_exp['Opex'] = opex
        dict_exp['Well Cost'] = well_cost
        dict_exp['Oil Price'] = oil_price
        dict_exp['Discount Rate'] = dis_rate
        dict_exp['Royalty Rate'] = royalty_rate
        dict_exp['Tax Rate'] = tax_rate
    return dict_exp


# In[ ]:


def Exp_profile(dict_prod,dict_exp,coef,intercept,spacing=0):
    """Returns a dataframe with Net Cash Flows & Proit 
     based on economic parameters and well spacing patterns"""
    import pandas as pd
    import numpy as np

    # To ignore warnings generated by the current version of seaborn 
    import warnings                  
    warnings.filterwarnings("ignore")  

    if (spacing==0):
        spacing = float(input("Enter Well Spacing for generating Economic Profile: "))
    profile_e = Production_profile(dict_prod,coef,intercept,spacing)

    year = (pd.Series([0.00])).append(profile_e['Year'],ignore_index=True)
    oil_prod = (pd.Series([0.00])).append(profile_e['Avg. Oil Rate, m3/d'],ignore_index=True)*(365/1000000)*6.2898
    gross_rev = oil_prod*dict_exp['Oil Price']
    opex = oil_prod*dict_exp['Opex']
    royalty = gross_rev*dict_exp['Royalty Rate']
    mwell = (pd.Series([0.00])).append(profile_e['Max Wells in Year'],ignore_index=True)
    w_cst = mwell*dict_exp['Well Cost']
    net_rev = gross_rev-w_cst-opex-royalty
    net_rev[0] = -1*dict_exp['Capex']
    tax = np.zeros(dict_prod['Years']+1)
    profit = np.zeros(dict_prod['Years']+1)
    cum_profit = np.zeros(dict_prod['Years']+1)
    for i in range(dict_prod['Years']+1):
        if (net_rev[i]<0):
            tax[i] = 0
        else:
            tax[i] = net_rev[i]*dict_exp['Tax Rate']
        profit[i] = net_rev[i]-tax[i]
        if (i==0):
            cum_profit[i] = profit[i]
        else:
            cum_profit[i] = profit[i]+cum_profit[i-1] 

    return pd.DataFrame(data={'Year':year,'Oil Production Rate, Mn bbls':oil_prod,'Gross Revenue, Mn $':gross_rev,
                     'OPEX, Mn $':opex,'Royalty, Mn $':royalty,'Well Cost, Mn $':w_cst,'Net Revenue, Mn $':net_rev,
                     'Tax, Mn $':tax,'Profit, Mn $':profit,'Cumulative Profit, Mn $':cum_profit})


# In[ ]:


def Eco_analysis_table(dict_exp,ep,spacing=0):
    """Returns a table with DCF, NPV, IRR & Payback Period 
     based on economic parameters and well spacing patterns"""
    import pandas as pd
    import numpy as np
    import numpy_financial as npf

    # To ignore warnings generated by the current version of seaborn 
    import warnings                  
    warnings.filterwarnings("ignore")  

    if (spacing==0):
        spacing = float(input("Enter Well Spacing for generating Master Table: "))
    # ep = exp_profile(coef,intercept,spacing)  
    m_table_c1 = ['Present Value of CAPEX (Mn $)','Present Worth of Revenue-Pre Tax (Mn $)','Present Worth of Revenue-Post Tax (Mn $)',
             'Net Present Value-Pre Tax (Mn $)','Net Present Value-Post Tax (Mn $)','Internal Rate of Return-Pre Tax (%)',
             'Internal Rate of Return-Post Tax (%)','Payback Period (Years)']#,'Break Even Oil Price ($/bbl)']
    m_table_c2 = np.zeros(len(m_table_c1))
    m_table_c2[0] = round(-1*dict_exp['Capex'],2)
    m_table_c2[1] = round(npf.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep['Net Revenue, Mn $'][1:],ignore_index=True)),2)
    m_table_c2[2] = round(npf.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep['Profit, Mn $'][1:],ignore_index=True)),2)
    m_table_c2[3] = m_table_c2[1]+m_table_c2[0]
    m_table_c2[4] = m_table_c2[2]+m_table_c2[0]
    m_table_c2[5] = round(npf.irr(ep['Net Revenue, Mn $'])*100,2)
    m_table_c2[6] = round(npf.irr(ep['Profit, Mn $'])*100,2)
    i = 0
    while (ep['Cumulative Profit, Mn $'][i]<0):
        i=i+1
        m_table_c2[7] = round((ep['Year'][i-1])-(ep['Cumulative Profit, Mn $'][i-1]/ep['Profit, Mn $'][i]),2)

    return pd.DataFrame(data={'Master Table':m_table_c1,str(spacing)+' m':m_table_c2})


# In[ ]:


def Eco_analysis(dict_prod,dict_exp,coef,intercept):
    """Returns a table with DCF, NPV, IRR & Payback Period 
     based on economic parameters for different well spacing patterns (700 m, 600 m,..., 200 m)"""
    import pandas as pd
    import numpy as np 

    ep_7 = Exp_profile(dict_prod,dict_exp,coef,intercept,700)
    ep_6 = Exp_profile(dict_prod,dict_exp,coef,intercept,600)
    ep_5 = Exp_profile(dict_prod,dict_exp,coef,intercept,500)
    ep_4 = Exp_profile(dict_prod,dict_exp,coef,intercept,400)
    ep_3 = Exp_profile(dict_prod,dict_exp,coef,intercept,300)
    ep_2 = Exp_profile(dict_prod,dict_exp,coef,intercept,200)
    data = pd.concat([Eco_analysis_table(dict_exp,ep_7,700),Eco_analysis_table(dict_exp,ep_6,600).iloc[:,1],
                    Eco_analysis_table(dict_exp,ep_5,500).iloc[:,1],Eco_analysis_table(dict_exp,ep_4,400).iloc[:,1],
                    Eco_analysis_table(dict_exp,ep_3,300).iloc[:,1],Eco_analysis_table(dict_exp,ep_2,200).iloc[:,1]],axis=1)
    npv_pt = data.iloc[4,1:].values.tolist()
    tmp = max(npv_pt)
    index = npv_pt.index(tmp)
    text = 'The best spacing pattern for this case would be '+str(data.columns[index+1])+' becuase this spacing pattern generates the highest NPV (Post Tax) equal to $'+str(round(max(npv_pt),2))+' Million and IRR (Post Tax) equal to '+str(data.iloc[6,index+1])+'%.'
    return data, text


# In[ ]:


def Cashflow_bar(dict_prod,dict_exp,coef,intercept,state,spacing=0):
    """Bargraph of Yearly Casflows for a well spacing patterns"""
    import plotly.express as px
    import plotly.graph_objects as go
    
    if (spacing==0):
        spacing = float(input("Enter Well Spacing for generating the bar graph: "))
    ep = Exp_profile(dict_prod,dict_exp,coef,intercept,spacing)
    
    fig = go.Figure(data=[
        go.Bar(x=ep['Year'],
               y=ep['Profit, Mn $'][1:],
               name=str(spacing)+' m',
               marker_color='lawngreen',
               showlegend=True)
    ])
    
    fig.update_layout(title='<b>Yearly Cashflows - '+state+'</b>',
                      title_x=0.5,
                      xaxis_title='Years',
                      yaxis_title='Net Profit, Mn $')
    return fig


# In[ ]:


def Cashflow_bar_clustered(dict_prod,dict_exp,coef1,intercept1,coef2,intercept2,spacing1=0,spacing2=0):
    """Clustered Bargraph of Yearly Casflows for well spacing pattern before and after EOR"""
    import plotly.express as px
    import plotly.graph_objects as go

    if (spacing1==0):
        spacing1 = float(input("Enter 1st Well Spacing for generating the bar graph: "))
    if (spacing2==0):
        spacing2 = float(input("Enter 2nd Well Spacing for generating the bar graph: "))
    ep1 = Exp_profile(dict_prod,dict_exp,coef1,intercept1,spacing1)
    ep2 = Exp_profile(dict_prod,dict_exp,coef2,intercept2,spacing2)

    fig = go.Figure(data=[
        go.Bar(x=ep1['Year'][1:]-0.2,
               y=ep1['Profit, Mn $'][1:],
               name='Before EOR - '+str(spacing1)+' m',
               marker_color='turquoise',
               showlegend=True),
        go.Bar(x=ep2['Year'][1:]+0.2,
               y=ep2['Profit, Mn $'][1:],
               name='After EOR - '+str(spacing2)+' m',
               marker_color='plum',
               showlegend=True)
    ])
    
    fig.update_layout(title='<b>Yearly Cashflows</b>',
                      title_x=0.5,
                      xaxis_title='Years',
                      yaxis_title='Net Profit, Mn $',
                      barmode='group',
                      bargap=0.15, # gap between bars of adjacent location coordinates.
                      bargroupgap=0.1, # gap between bars of the same location coordinate.
                      legend=dict(
                          x=0,
                          y=1.0,
                          bgcolor='rgba(255, 255, 255, 0)',
                          bordercolor='rgba(255, 255, 255, 0)'))
    return fig    


# In[ ]:


def Sensitivity(dict_prod,dict_exp,coef,intercept,spacing=0):
    """Returns NPV for +10%/-10% change in CAPEX, OPEX, Oil Price, Discount Rate 
     and Tax Rate for a well spacing pattern"""
    import pandas as pd
    import numpy as np
    import numpy_financial as npf

    # To ignore warnings generated by the current version of seaborn 
    import warnings                  
    warnings.filterwarnings("ignore")  

    if (spacing==0):
        spacing = float(input("Enter Well Spacing for generating Economic Profile: "))

    capex,opex,well_cost,oil_price,dis_rate,royalty_rate,tax_rate = dict_exp['Capex'],dict_exp['Opex'],dict_exp['Well Cost'],dict_exp['Oil Price'],dict_exp['Discount Rate'],dict_exp['Royalty Rate'],dict_exp['Tax Rate']
    ep = Exp_profile(dict_prod,dict_exp,coef,intercept,spacing)
    npv_present = round(-1*dict_exp['Capex'],2)+round(npf.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep['Profit, Mn $'][1:],ignore_index=True)),2)

    ep90_capex = Exp_profile(dict_prod,Exp_dictionary(capex*0.9,opex,well_cost,oil_price,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
    ep90_opex = Exp_profile(dict_prod,Exp_dictionary(capex,opex*0.9,well_cost,oil_price,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
    ep90_oilprice = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price*0.9,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
    ep90_disrate = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price,dis_rate*0.9,royalty_rate,tax_rate),coef,intercept,spacing)
    ep90_taxrate = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price,dis_rate,royalty_rate,tax_rate*0.9),coef,intercept,spacing)

    npv_90 = np.zeros(5)
    npv_90[0] = round(-1*(capex),2)+round(npf.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep90_oilprice['Profit, Mn $'][1:],ignore_index=True)),2) 
    npv_90[1] = round(-1*(capex*0.9),2)+round(npf.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep90_capex['Profit, Mn $'][1:],ignore_index=True)),2)
    npv_90[2] = round(-1*(capex),2)+round(npf.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep90_opex['Profit, Mn $'][1:],ignore_index=True)),2)
    npv_90[3] = round(-1*(capex),2)+round(npf.npv((dis_rate*0.9),(pd.Series([0.00])).append(ep90_disrate['Profit, Mn $'][1:],ignore_index=True)),2)
    npv_90[4] = round(-1*(capex),2)+round(npf.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep90_taxrate['Profit, Mn $'][1:],ignore_index=True)),2)

    ep110_capex = Exp_profile(dict_prod,Exp_dictionary(capex*1.1,opex,well_cost,oil_price,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
    ep110_opex = Exp_profile(dict_prod,Exp_dictionary(capex,opex*1.1,well_cost,oil_price,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
    ep110_oilprice = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price*1.1,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
    ep110_disrate = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price,dis_rate*1.1,royalty_rate,tax_rate),coef,intercept,spacing)
    ep110_taxrate = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price,dis_rate,royalty_rate,tax_rate*1.1),coef,intercept,spacing)

    npv_110 = np.zeros(5)
    npv_110[0] = round(-1*(capex),2)+round(npf.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep110_oilprice['Profit, Mn $'][1:],ignore_index=True)),2)
    npv_110[1] = round(-1*(capex*1.1),2)+round(npf.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep110_capex['Profit, Mn $'][1:],ignore_index=True)),2)
    npv_110[2] = round(-1*(capex),2)+round(npf.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep110_opex['Profit, Mn $'][1:],ignore_index=True)),2)
    npv_110[3] = round(-1*(capex),2)+round(npf.npv((dis_rate*1.1),(pd.Series([0.00])).append(ep110_disrate['Profit, Mn $'][1:],ignore_index=True)),2)
    npv_110[4] = round(-1*(capex),2)+round(npf.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep110_taxrate['Profit, Mn $'][1:],ignore_index=True)),2)

    return npv_present, npv_90, npv_110


# In[ ]:


def Sensitivity_chart(dict_prod,dict_exp,coef,intercept,state,spacing=0):
    """Tornado Chart and Spider Plot for change in Economic Parameters vs  
     Sensitivity of NPV for a well spacing pattern"""
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    npv_present, npv_90, npv_110 = Sensitivity(dict_prod,dict_exp,coef,intercept,spacing)
    ylab = np.array(['Oil Price', 'CAPEX', 'OPEX', 'Discount Rate', 'Tax'])
    x_90 = npv_90 - npv_present 
    x_110 = npv_110 - npv_present

    x = np.arange(-0.1,0.11,0.1)
    y_90 = (npv_90/npv_present) - 1
    y_110 = (npv_110/npv_present) - 1 

    
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Tornado Chart','Sensitivity Spider Plot'))
    
    # Tornado Chart
    fig.add_trace(go.Bar(x=x_90,
                         y=ylab,
                         width=0.5,
                         name='-10% Change',
                         orientation='h',
                         marker=dict(
                             color='palegreen',
                             line=dict(color='black', width=0.5))),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=x_110,
                         y=ylab,
                         width=0.5,
                         name='+10% Change',
                         orientation='h',
                         marker=dict(
                             color='plum',
                             line=dict(color='black', width=0.5))),
                  row=1, col=1)
    fig.add_vline(x=0,
                  line_width=2,
                  line_color="black",
                  row=1, col=1)
    fig.update_layout(barmode='overlay',
                      bargap=0.30)
    
    # Sensitivity Spider Plot
    fig.add_trace(go.Scatter(x=x,
                             y=[y_90[0],0,y_110[0]],
                             mode='lines+markers',
                             name='Oil Price',
                             line=dict(color='limegreen', width=4),
                             showlegend=True),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x,
                             y=[y_90[1],0,y_110[1]],
                             mode='lines+markers',
                             name='CAPEX',
                             line=dict(color='red', width=4),
                             showlegend=True),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x,
                             y=[y_90[2],0,y_110[2]],
                             mode='lines+markers',
                             name='OPEX',
                             line=dict(color='indigo', width=4),
                             showlegend=True),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x,
                             y=[y_90[3],0,y_110[3]],
                             mode='lines+markers',
                             name='Discount Rate',
                             line=dict(color='yellow', width=4),
                             showlegend=True),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x,
                             y=[y_90[4],0,y_110[4]],
                             mode='lines+markers',
                             name='Tax',
                             line=dict(color='deepskyblue', width=4),
                             showlegend=True),
                  row=1, col=2)
    fig.add_vline(x=0, 
                  line_width=2,
                  line_dash="dash",
                  line_color="gray",
                  row=1, col=2)
    fig.add_hline(y=0,
                  line_width=2,
                  line_dash="dash",
                  line_color="gray",
                  row=1, col=2)
    
    # Update x axis properties
    fig.update_xaxes(title_text="Net Present Value (Post Tax), Mn $", row=1, col=1)
    fig.update_xaxes(title_text="Percentage Change", row=1, col=2)
    

    fig.update_yaxes(title_text="Net Present Value (Post Tax)", title_standoff = 0, row=1, col=2)
    
    fig.update_layout(title='<b>'+state+'</b>',
                      title_x=0.5)
    
    return fig

