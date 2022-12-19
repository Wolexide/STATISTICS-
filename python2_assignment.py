# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:56:41 2022

@author: user
"""
# import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# define a function to extract the data and the transposed dataframe
def extracted_data(file_path, countries, columns):
    """
    The necessary parameters are
    file_path : this is the url of the world bank data
    countries : list of countries to be analysed.
    columns : list of years to be used.
    """
    df = pd.read_excel(file_path, sheet_name= 'Data', skiprows=3)
    df = df[columns]
    df.set_index('Country Name', inplace = True)
    df = df.loc[countries]
    return df, df.transpose()

# the file paths of the indicators are stored in the variables below
path_1 = 'https://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.PC?downloadformat=excel'

path_2 = 'https://api.worldbank.org/v2/en/indicator/SH.DYN.MORT?downloadformat=excel'

path_3 = 'https://api.worldbank.org/v2/en/indicator/EG.USE.ELEC.KH.PC?downloadformat=excel'

# Five countries are considered for the visualizations
countries = ['Belgium','Bulgaria','Cuba','Finland', 'Nigeria']

# Years range from 2000 to 2010.
columns = ['Country Name', '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010']

# the parameters for the functions were passed and it is used to produce various plots for the analysis
year_C02, country_C02 = extracted_data(path_1, countries, columns)
year_mortality, country_mortality = extracted_data(path_2, countries, columns)
year_elect_power, country_elect_power,  = extracted_data(path_3, countries, columns)

#create a discriptive statistics for electricity power consumption
describe_elect_power = country_elect_power.describe()
print(describe_elect_power)



# matplotlib is used to create a multiple plot of Mortality countries
plt.figure(figsize=(9,8),dpi=200)
for i in range(len(countries)):
    #the values on the x-axis is plot using the index while the countries is plot on the y-axis
    plt.plot(country_mortality.index, country_mortality[countries[i]],label=countries[i]) 
plt.legend(bbox_to_anchor=(1.2,1.01))
plt.title('Mortality rate of five countries over ten years', fontweight='bold')
plt.xlabel('Year', fontsize=20)
plt.ylabel('Rate', fontsize=20)
plt.show()
plt.savefig('mortality_rate.png')

# matplotlib is used to create a multiple plot of Electricity power consumption
plt.figure(figsize=(9,8),dpi=200)
for i in range(len(countries)):
    #the values on the x-axis is plot using the index while the countries are what will be plotted on the y-axis
    plt.plot(country_elect_power.index, country_elect_power[countries[i]], label=countries[i]) 
plt.legend(bbox_to_anchor=(1.2,1.01))
plt.title('Electricity power consumption of five countries over 10 years', fontweight='bold')
plt.xlabel('Year', fontsize=20)
plt.ylabel('kWh per Capita', fontsize=20)
plt.savefig('Electric_power.png')
plt.show()

# the parameters below are used for the multiple bar plots below
array_label = ['Belgium','Bulgaria','Cuba','Finland','Nigeria']
width = 0.2
x_values = np.arange(len(array_label)) # this is the length of array_label

# matplotlib is used for defining the multiple bar plots of C02 Emissions from 2000 to 2010 with increments of 5 years
fig, ax  = plt.subplots(figsize=(12,8), dpi=200) # this sets the figsize of the bar plots

plt.bar(x_values - width, year_C02['2000'], width, label='Year 2000') # this dictates the size of the plots
plt.bar(x_values, year_C02['2005'], width, label='Year 2005')
plt.bar(x_values + width, year_C02['2010'], width, label='Year 2010')
    
    
plt.title('Multiple bar plots showing C02 Emissions across five countries ', fontsize=20, fontweight='bold')
plt.ylabel('Metric tons', fontsize=20)
plt.xticks(x_values, array_label)

plt.legend()

plt.show() # the multiple bar plot is displayed below

# matplotlib is used for defining the multiple bar plots of Mortality rate from 2000 to 2010 with increments of 5 years
fig, ax  = plt.subplots(figsize=(12,8), dpi=200)  # this sets the figsize of the bar plots

plt.bar(x_values - width, year_mortality['2000'], width, label='Year 2000') 
plt.bar(x_values, year_mortality['2005'], width, label='Year 2005')
plt.bar(x_values + width, year_mortality['2010'], width, label='Year 2010')
    
    
plt.title('Multiple bar plots showing mortality rate across different countries', fontsize=20, fontweight='bold')
plt.ylabel('Mortality rate', fontsize=20)
plt.xticks(x_values, array_label)


plt.legend()

plt.show() # the multiple bar plot is displayed below

# a dataframe is created using Nigeria which takes 3 indicators as parameters
df_Nigeria = pd.DataFrame({'C02 Emissions': country_C02['Nigeria'],
        'Mortality rate': country_mortality['Nigeria'],
        'Electricity power consumption': country_elect_power['Nigeria']        
        })
print(df_Nigeria)

# a dataframe is created using Finland which takes 3 indicators as parameters
df_Finland = pd.DataFrame({'C02 Emissions': country_C02['Finland'],
        'Mortality rate': country_mortality['Finland'],
        'Electricity power consumption': country_elect_power['Finland']        
        })
print(df_Finland)

# the correlation matrix of the dataframe is examined using the .corr 
Nigeria_corr = df_Nigeria.corr()
print(Nigeria_corr)

# the correlation matrix of the dataframe is examined using the .corr 
Finland_corr = df_Finland.corr()
print(Finland_corr)

# the function creates a heatmap
def heatmap(corr_matrix, title):
    """ This function defines a heatmap that accept the correlation matrix and title of the map as parameters """
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.0)
    sns.heatmap(corr_matrix, annot=True) # seaborn is used to produce the heatmap of the correlation matrix of the indicators
    plt.title(title, fontsize=20, fontweight='bold')
    return

# the heatmap of Nigeria is displayed below
heatmap(Nigeria_corr, 'Nigeria')

# the heatmap of Finland is displayed below
heatmap(Finland_corr, 'Finland')