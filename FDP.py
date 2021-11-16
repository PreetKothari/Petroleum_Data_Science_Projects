"""
Modelling Field Development Model

Objectives: 
* Model Waterflood Recovery and Recovery after EOR based on given Swi, Sor,
  muo, and muw. 
* Comparision between EOR & Waterflood Recovery.
* Generating Relative Permeablity and Fractional Flow Curves.
* Generating Production Profiles over a time period for given Area, OIIP, 
  Test Liquid Rate and Well Spacing pattern.
* Basic Economic Analysis based on different Production Profiles.
* Generating Petroleum Project Net Cash Flows (NCF), Net Present Value (NPV),
  Internal Rate of Return (IRR) and Different plots for in-depth Economic 
  Analysis based on given CAPEX, OPEX, Tax, Royalty, and Oil Prices.
* Sensitivity Analysis through Tornado Charts and Spider Plots.

@author: Preet Kothari
@email: preetkothari19@gmail.com

Note: 
Depreciation is still to be included.
"""

"""### **Functions** """

""" #### Recovery Analysis Functions - Waterflood & EOR """

def Rec_Analysis():
  """Collects values of Water Saturation (Swi) """
  Swi = float(input("Enter Initial Water Saturation, Swi: "))
  Soi = 1-Swi
  print('Initial Oil Saturation, Soi:',Soi)
  Sorw = float(input("Enter Residual Oil Saturation to Water, Sorw: "))
  muo = float(input("Enter Oil Viscosity, Muo: "))
  muw = float(input("Enter Water Viscosity, Muw: "))
  
  R = np.arange(0,0.61,0.01)        # Recovery
  Sw = R*(1-Swi) + Swi              # Water Saturation
  So = (1-Sw)                       # Oil Saturation
  Sw_norm = (Sw-Swi)/(1-Swi-Sorw)   # Normalized Water Saturation
  Krw = (Sw_norm)**3                # Relative Permeabiltiy to Water
  Kro = (1-Sw_norm)**3              # Relative Permeabiltiy to Oil
  M = (Krw*muo)/(Kro*muw)           # Mobility Contrast
  Fw = 1/(1+(1/M))                  # Fractional Water Cut
  
  data = pd.DataFrame(data={'Recovery':R,'Oil Saturation':So,'Water Saturation':Sw,'Normalized Water Saturation':Sw_norm,'Water Rel Perm':Krw,'Oil Rel Perm':Kro,'Mobility Contrast':M,'Water Cut':Fw})
  return data

help(Rec_Analysis)

def Rel_Perm_Curve(data):
    
  # Relative Permeability Curve
  plt.plot(data['Water Saturation'],data['Water Rel Perm'],label='Krw')
  plt.plot(data['Water Saturation'],data['Oil Rel Perm'],label='Kro')
  plt.plot(np.full((data.shape[0]),0.5),data['Oil Rel Perm'],color='black',linestyle='dashed', linewidth = 1)
  plt.xlabel('Water Saturation, Sw')
  plt.xlim(0,0.8)
  plt.ylabel('Rel Perm, Krw & Kro')
  plt.ylim(0,1)
  plt.title('Relative Permeability Curve')
  plt.legend()

  plt.show()

def Curves(data):
  f,ax = plt.subplots(1,2,figsize=(22,6))

  # Fractional Flow Curve
  ax[0].plot(data['Water Saturation'],data['Water Cut'],color='purple')
  ax[0].set_xlabel('Water Saturation, Sw')
  ax[0].set_xlim(0,0.8)
  ax[0].set_ylabel('Water Cut, Fw')
  ax[0].set_ylim(0,1)
  ax[0].set_title('Fractional Flow Curve')

  # Recovery vs Water Cut
  ax[1].plot(data['Recovery'],data['Water Cut'],color='limegreen')
  ax[1].set_xlabel('Recovery, R')
  ax[1].set_xlim(0,0.8)
  ax[1].set_ylabel('Water Cut, Fw')
  ax[1].set_ylim(0,1)
  ax[1].set_title('Recovery vs Water Cut')

  plt.show()

def EOR_Curves(data_bf_eor,data_af_eor):
  r_coef_bf_eor,r_intercept_bf_eor = Poly_eq(6,data_bf_eor.iloc[:,0:1].values,data_bf_eor.iloc[:,-1:].values)
  r_coef_af_eor,r_intercept_af_eor = Poly_eq(6,data_af_eor.iloc[:,0:1].values,data_af_eor.iloc[:,-1:].values)
  sw_coef_bf_eor,sw_intercept_bf_eor = Poly_eq(6,data_bf_eor.iloc[:,2:3].values,data_bf_eor.iloc[:,-1:].values)
  sw_coef_af_eor,sw_intercept_af_eor = Poly_eq(6,data_af_eor.iloc[:,2:3].values,data_af_eor.iloc[:,-1:].values)
 
  fwr_bf = round(Sol(r_coef_bf_eor,r_intercept_bf_eor),3)
  fwr_af = round(Sol(r_coef_af_eor,r_intercept_af_eor),3)
  fws_bf = round(Sol(sw_coef_bf_eor,sw_intercept_bf_eor),3)
  fws_af = round(Sol(sw_coef_af_eor,sw_intercept_af_eor),3)

  f,ax = plt.subplots(1,2,figsize=(22,6))

  # Fractional Flow Curve - Before and After EOR
  ax[0].plot(data_bf_eor['Water Saturation'],data_bf_eor['Water Cut'],color='purple',label='Before EOR',marker='o')
  ax[0].plot(data_af_eor['Water Saturation'],data_af_eor['Water Cut'],color='deepskyblue',label='After EOR',marker='o')
  ax[0].plot(np.arange(0,(fws_af+0.01),fws_af),np.full(2,0.9),color='black',linestyle='dashed', linewidth = 1)
  ax[0].plot(np.full(2,fws_af),np.arange(0,0.91,0.9),color='black',linestyle='dashed', linewidth = 1)
  ax[0].plot(np.full(2,fws_bf),np.arange(0,0.91,0.9),color='black',linestyle='dashed', linewidth = 1)
  ax[0].set_xlabel('Water Saturation, Sw')
  ax[0].set_xlim(0,0.8)
  ax[0].set_ylabel('Water Cut, Fw')
  ax[0].set_ylim(0,1)
  ax[0].set_title('Fractional Flow Curve')
  ax[0].legend()

  # Recovery vs Water Cut - Before and After EOR
  ax[1].plot(data_bf_eor['Recovery'],data_bf_eor['Water Cut'],color='limegreen',label='Before EOR',marker='o')
  ax[1].plot(data_af_eor['Recovery'],data_af_eor['Water Cut'],color='red',label='After EOR',marker='o')
  ax[1].plot(np.arange(0,(fwr_af+0.01),fwr_af),np.full(2,0.9),color='black',linestyle='dashed', linewidth = 1)
  ax[1].plot(np.full(2,fwr_af),np.arange(0,0.91,0.9),color='black',linestyle='dashed', linewidth = 1)
  ax[1].plot(np.full(2,fwr_bf),np.arange(0,0.91,0.9),color='black',linestyle='dashed', linewidth = 1)
  ax[1].set_xlabel('Recovery, R')
  ax[1].set_xlim(0,0.8)
  ax[1].set_ylabel('Water Cut, Fw')
  ax[1].set_ylim(0,1)
  ax[1].set_title('Recovery vs Water Cut')
  ax[1].legend()
  plt.show()

  # rec_bf_eor = wf_data[wf_data['Water Cut']==min(wf_data['Water Cut'][wf_data['Water Cut']>0.90])].iloc[0]['Recovery']
  # sat_bf_eor = wf_data[wf_data['Water Cut']==min(wf_data['Water Cut'][wf_data['Water Cut']>0.90])].iloc[0]['Water Saturation']
  # rec_af_eor = eor_data[eor_data['Water Cut']==min(eor_data['Water Cut'][eor_data['Water Cut']>0.90])].iloc[0]['Recovery']
  # sat_af_eor = eor_data[eor_data['Water Cut']==min(eor_data['Water Cut'][eor_data['Water Cut']>0.90])].iloc[0]['Water Saturation']

  print('The approximate recovery at 90% water cut before EOR was {0}% and after applying EOR the approximate recovery was {1}%.'.format(int(fwr_bf*100),int(fwr_af*100))) 
  print('The approximate water saturation at 90% water cut before EOR was {0}% and after applying EOR the approximate water saturation was {1}%.'.format(int(fws_bf*100),int(fws_af*100)))

"""#### Functions for Generating 6 Degree Polynomial Equations and their Solutions"""

def Poly_eq(degree,x,y):
  # import operator
  # from sklearn.linear_model import LinearRegression
  # from sklearn.metrics import mean_squared_error, r2_score
  # from sklearn.preprocessing import PolynomialFeatures

  # x = wf_data['Recovery'].values
  # y = wf_data['Water Cut'].values

  # x = x[:, np.newaxis]
  # y = y[:, np.newaxis]

  # x = wf_data.iloc[:,0:1].values
  # y = wf_data.iloc[:,-1:].values

  polynomial_features= PolynomialFeatures(degree=6)
  x_poly = polynomial_features.fit_transform(x)

  model = LinearRegression()
  model.fit(x_poly, y)
  y_poly_pred = model.predict(x_poly)

  # rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
  # r2 = r2_score(y,y_poly_pred)
  # print(rmse)
  # print(r2)

  # plt.scatter(x, y)

  #sorting predicted values with respect to predictor
  # sort_axis = operator.itemgetter(0)
  # sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
  # x_poly, y_poly_pred = zip(*sorted_zip)

  # plt.plot(x, y_poly_pred, '--', color='m',linewidth=2)
  # plt.show()

  return model.coef_, model.intercept_

def Sol(coef,intercept):
  x = symbols('x')
  val = solve((coef[0][6]*(x**6)) + (coef[0][5]*(x**5)) + (coef[0][4]*(x**4)) + (coef[0][3]*(x**3)) + (coef[0][2]*(x**2)) + (coef[0][1]*x) + (intercept[0]) - 0.9, x)
  return val[1]

def Fractional_water_cut(x,coef,intercept):
  return (coef[0][6]*(x**6)) + (coef[0][5]*(x**5)) + (coef[0][4]*(x**4)) + (coef[0][3]*(x**3)) + (coef[0][2]*(x**2)) + (coef[0][1]*x) + (intercept)

"""#### Production Profile Functions"""

def Prod_dictionary(A=0,Oiip=0,Years=25,test_liq_rate=0):
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

def Well_req(dict_prod,spacing):
  spacing = float(spacing)/1000
  return int((dict_prod['Area']*0.9)/(spacing**2))

def Drilled_wells(w,dict_prod):
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

def Production_profile(dict_prod,coef,intercept,spacing=0):
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

def Avg_rate_stacked_bar(profile):
  plt.bar('Year','Avg. Oil Rate, m3/d',data=profile,label='Avg. Oil Rate',color='orangered')
  plt.bar('Year','Avg. Water Rate, m3/d',data=profile,bottom='Avg. Oil Rate, m3/d',label='Avg. Water Rate')
  plt.xlabel('Years')
  plt.ylabel('Flow Rate, m3/d')
  plt.title('Year-wise Average Oil and Water Flow Rate')
  plt.legend(loc=4)
  plt.show()

"""#### Economical Analysis Functions"""

def Eco_curves(dict_prod,coef,intercept):
     
  profile_7 = Production_profile(dict_prod,coef,intercept,700)
  profile_6 = Production_profile(dict_prod,coef,intercept,600)
  profile_5 = Production_profile(dict_prod,coef,intercept,500)
  profile_4 = Production_profile(dict_prod,coef,intercept,400)
  profile_3 = Production_profile(dict_prod,coef,intercept,300)
  profile_2 = Production_profile(dict_prod,coef,intercept,200)
  
  f,ax = plt.subplots(1,3,figsize=(26,8))

  # Oil Rate vs Year
  ax[0].plot(profile_7['Year'],profile_7['Avg. Oil Rate, m3/d'],label='700 m',color='indigo')
  ax[0].plot(profile_6['Year'],profile_6['Avg. Oil Rate, m3/d'],label='600 m',color='deepskyblue')
  ax[0].plot(profile_5['Year'],profile_5['Avg. Oil Rate, m3/d'],label='500 m',color='limegreen')
  ax[0].plot(profile_4['Year'],profile_4['Avg. Oil Rate, m3/d'],label='400 m',color='yellow')
  ax[0].plot(profile_3['Year'],profile_3['Avg. Oil Rate, m3/d'],label='300 m',color='orange')
  ax[0].plot(profile_2['Year'],profile_2['Avg. Oil Rate, m3/d'],label='200 m',color='red')
  ax[0].set_xlabel('Year')
  ax[0].set_xlim(0,25)
  ax[0].set_ylabel('Oil Rate, m3/d')
  ax[0].set_title('Yearly Oil Rate based on Well Spacing')
  ax[0].legend()

  # Recovery vs Year
  ax[1].plot(profile_7['Year'],profile_7['Recovery, %'],label='700 m',color='indigo')
  ax[1].plot(profile_6['Year'],profile_6['Recovery, %'],label='600 m',color='deepskyblue')
  ax[1].plot(profile_5['Year'],profile_5['Recovery, %'],label='500 m',color='limegreen')
  ax[1].plot(profile_4['Year'],profile_4['Recovery, %'],label='400 m',color='yellow')
  ax[1].plot(profile_3['Year'],profile_3['Recovery, %'],label='300 m',color='orange')
  ax[1].plot(profile_2['Year'],profile_2['Recovery, %'],label='200 m',color='red')
  ax[1].set_xlabel('Year')
  ax[1].set_xlim(0,25)
  ax[1].set_ylabel('Recovery, %')
  ax[1].set_title('Yearly Recovery based on Well Spacing')
  ax[1].legend()

  # Avg. Water Cut vs Year
  ax[2].plot(profile_7['Year'],profile_7['Avg. Water-Cut, %'],label='700 m',color='indigo')
  ax[2].plot(profile_6['Year'],profile_6['Avg. Water-Cut, %'],label='600 m',color='deepskyblue')
  ax[2].plot(profile_5['Year'],profile_5['Avg. Water-Cut, %'],label='500 m',color='limegreen')
  ax[2].plot(profile_4['Year'],profile_4['Avg. Water-Cut, %'],label='400 m',color='yellow')
  ax[2].plot(profile_3['Year'],profile_3['Avg. Water-Cut, %'],label='300 m',color='orange')
  ax[2].plot(profile_2['Year'],profile_2['Avg. Water-Cut, %'],label='200 m',color='red')
  ax[2].set_xlabel('Year')
  ax[2].set_xlim(0,25)
  ax[2].set_ylabel('Avg. Water Cut, %')
  ax[2].set_title('Yearly Avg. Water Cut based on Well Spacing')
  ax[2].legend()

  plt.show()

def Exp_dictionary(capex=0,opex=0,well_cost=0,oil_price=0,dis_rate=0,royalty_rate=0,tax_rate=0):
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

def Exp_profile(dict_prod,dict_exp,coef,intercept,spacing=0):
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

def Eco_analysis_table(dict_exp,ep,spacing=0):
  if (spacing==0):
    spacing = float(input("Enter Well Spacing for generating Master Table: "))
  # ep = exp_profile(coef,intercept,spacing)  
  m_table_c1 = ['Present Value of CAPEX (Mn $)','Present Worth of Revenue-Pre Tax (Mn $)','Present Worth of Revenue-Post Tax (Mn $)',
             'Net Present Value-Pre Tax (Mn $)','Net Present Value-Post Tax (Mn $)','Internal Rate of Return-Pre Tax (%)',
             'Internal Rate of Return-Post Tax (%)','Payback Period (Years)']#,'Break Even Oil Price ($/bbl)']
  m_table_c2 = np.zeros(len(m_table_c1))
  m_table_c2[0] = round(-1*dict_exp['Capex'],2)
  m_table_c2[1] = round(np.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep['Net Revenue, Mn $'][1:],ignore_index=True)),2)
  m_table_c2[2] = round(np.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep['Profit, Mn $'][1:],ignore_index=True)),2)
  m_table_c2[3] = m_table_c2[1]+m_table_c2[0]
  m_table_c2[4] = m_table_c2[2]+m_table_c2[0]
  m_table_c2[5] = round(np.irr(ep['Net Revenue, Mn $'])*100,2)
  m_table_c2[6] = round(np.irr(ep['Profit, Mn $'])*100,2)
  i = 0
  while (ep['Cumulative Profit, Mn $'][i]<0):
    i=i+1
  m_table_c2[7] = round((ep['Year'][i-1])-(ep['Cumulative Profit, Mn $'][i-1]/ep['Profit, Mn $'][i]),2)
  
  return pd.DataFrame(data={'Master Table':m_table_c1,str(spacing)+' m':m_table_c2})

def Eco_analysis(dict_prod,dict_exp,coef,intercept):
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
  print('The best spacing pattern for this case would be {0} becuase this spacing pattern generates the highest NPV (Post Tax) equal to ${1} Million and IRR (Post Tax) equal to {2}%.'.format(data.columns[index+1],round(max(npv_pt),2),data.iloc[6,index+1]))
  return data

def Cashflow_bar(dict_prod,dict_exp,coef,intercept,spacing=0):
  if (spacing==0):
    spacing = float(input("Enter Well Spacing for generating the bar graph: "))
  ep = Exp_profile(dict_prod,dict_exp,coef,intercept,spacing)
  plt.bar(ep['Year'][1:],ep['Profit, Mn $'][1:],color='lawngreen',label=str(spacing)+' m')
  plt.xlabel('Years')
  plt.ylabel('Net Profit, Mn $')
  plt.title('Yearly Cashflows')
  plt.legend()
  plt.show()

def Cashflow_bar_clustered(dict_prod,dict_exp,coef1,intercept1,coef2,intercept2,spacing1=0,spacing2=0):
  if (spacing1==0):
    spacing1 = float(input("Enter 1st Well Spacing for generating the bar graph: "))
  if (spacing2==0):
    spacing2 = float(input("Enter 2nd Well Spacing for generating the bar graph: "))
  ep1 = Exp_profile(dict_prod,dict_exp,coef1,intercept1,spacing1)
  ep2 = Exp_profile(dict_prod,dict_exp,coef2,intercept2,spacing2)

  width = 0.40
  plt.bar(ep1['Year'][1:]-0.2,ep1['Profit, Mn $'][1:],color='turquoise',label='Before EOR - '+str(spacing1)+' m',width=width)
  plt.bar(ep2['Year'][1:]+0.2,ep2['Profit, Mn $'][1:],color='plum',label='After EOR - '+str(spacing2)+' m',width=width)
  plt.xlabel('Years')
  plt.ylabel('Net Profit, Mn $')
  plt.title('Yearly Cashflows')
  plt.legend()
  plt.show()

def Sensitivity(dict_prod,dict_exp,coef,intercept,spacing=0):
  if (spacing==0):
    spacing = float(input("Enter Well Spacing for generating Economic Profile: "))
  
  capex,opex,well_cost,oil_price,dis_rate,royalty_rate,tax_rate = dict_exp['Capex'],dict_exp['Opex'],dict_exp['Well Cost'],dict_exp['Oil Price'],dict_exp['Discount Rate'],dict_exp['Royalty Rate'],dict_exp['Tax Rate']
  ep = Exp_profile(dict_prod,dict_exp,coef,intercept,spacing)
  npv_present = round(-1*dict_exp['Capex'],2)+round(np.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep['Profit, Mn $'][1:],ignore_index=True)),2)

  ep90_capex = Exp_profile(dict_prod,Exp_dictionary(capex*0.9,opex,well_cost,oil_price,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
  ep90_opex = Exp_profile(dict_prod,Exp_dictionary(capex,opex*0.9,well_cost,oil_price,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
  ep90_oilprice = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price*0.9,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
  ep90_disrate = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price,dis_rate*0.9,royalty_rate,tax_rate),coef,intercept,spacing)
  ep90_taxrate = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price,dis_rate,royalty_rate,tax_rate*0.9),coef,intercept,spacing)
  
  npv_90 = np.zeros(5)
  npv_90[0] = round(-1*(capex),2)+round(np.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep90_oilprice['Profit, Mn $'][1:],ignore_index=True)),2) 
  npv_90[1] = round(-1*(capex*0.9),2)+round(np.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep90_capex['Profit, Mn $'][1:],ignore_index=True)),2)
  npv_90[2] = round(-1*(capex),2)+round(np.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep90_opex['Profit, Mn $'][1:],ignore_index=True)),2)
  npv_90[3] = round(-1*(capex),2)+round(np.npv((dis_rate*0.9),(pd.Series([0.00])).append(ep90_disrate['Profit, Mn $'][1:],ignore_index=True)),2)
  npv_90[4] = round(-1*(capex),2)+round(np.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep90_taxrate['Profit, Mn $'][1:],ignore_index=True)),2)
  
  ep110_capex = Exp_profile(dict_prod,Exp_dictionary(capex*1.1,opex,well_cost,oil_price,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
  ep110_opex = Exp_profile(dict_prod,Exp_dictionary(capex,opex*1.1,well_cost,oil_price,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
  ep110_oilprice = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price*1.1,dis_rate,royalty_rate,tax_rate),coef,intercept,spacing)
  ep110_disrate = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price,dis_rate*1.1,royalty_rate,tax_rate),coef,intercept,spacing)
  ep110_taxrate = Exp_profile(dict_prod,Exp_dictionary(capex,opex,well_cost,oil_price,dis_rate,royalty_rate,tax_rate*1.1),coef,intercept,spacing)
  
  npv_110 = np.zeros(5)
  npv_110[0] = round(-1*(capex),2)+round(np.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep110_oilprice['Profit, Mn $'][1:],ignore_index=True)),2)
  npv_110[1] = round(-1*(capex*1.1),2)+round(np.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep110_capex['Profit, Mn $'][1:],ignore_index=True)),2)
  npv_110[2] = round(-1*(capex),2)+round(np.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep110_opex['Profit, Mn $'][1:],ignore_index=True)),2)
  npv_110[3] = round(-1*(capex),2)+round(np.npv((dis_rate*1.1),(pd.Series([0.00])).append(ep110_disrate['Profit, Mn $'][1:],ignore_index=True)),2)
  npv_110[4] = round(-1*(capex),2)+round(np.npv(dict_exp['Discount Rate'],(pd.Series([0.00])).append(ep110_taxrate['Profit, Mn $'][1:],ignore_index=True)),2)
 
  return npv_present, npv_90, npv_110

def Sensitivity_chart(dict_prod,dict_exp,coef,intercept,spacing=0):
  npv_present, npv_90, npv_110 = Sensitivity(dict_prod,dict_exp,coef,intercept,spacing)
  ylab = np.array(['Oil Price', 'CAPEX', 'OPEX', 'Discount Rate', 'Tax'])
  x_90 = npv_90 - npv_present 
  x_110 = npv_110 - npv_present
  pos = np.arange(len(ylab)) + .5 # bars centered on the y axis
  
  x = np.arange(-0.1,0.11,0.1)
  y_90 = (npv_90/npv_present) - 1
  y_110 = (npv_110/npv_present) - 1 

  fig, ax = plt.subplots(ncols=2, figsize=plt.figaspect(1/3)) # aspect: three times as wide as tall

  ax[0].barh(pos, x_90, color='palegreen',label='-10%')
  ax[0].barh(pos, x_110, facecolor='plum',label='+10%')
  ax[0].set_yticks(pos)
  ax[0].set_yticklabels(ylab, ha='right')
  ax[0].axvline(0, color='gray', linewidth=3)
  ax[0].set_xlabel('Net Present Value (Post Tax), Mn $')
  ax[0].set_title('Tornado Chart')
  ax[0].legend()

  ax[1].plot(x,(y_90[0],0,y_110[0]),label='Oil Price',color='limegreen',marker='o')
  ax[1].plot(x,(y_90[1],0,y_110[1]),label='CAPEX',color='red',marker='o')
  ax[1].plot(x,(y_90[2],0,y_110[2]),label='OPEX',color='indigo',marker='o')
  ax[1].plot(x,(y_90[3],0,y_110[3]),label='Discount Rate',color='yellow',marker='o')
  ax[1].plot(x,(y_90[4],0,y_110[4]),label='Tax',color='deepskyblue',marker='o')
  ax[1].axvline(0, color='gray', linewidth=1.5)
  ax[1].axhline(0, color='gray', linewidth=1.5)
  ax[1].set_ylabel('Net Present Value (Post Tax)')
  ax[1].set_xlabel('Percentage Change')
  ax[1].set_title('Sensitivity Spider Plot')
  ax[1].legend()

  plt.show()
