import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex
import biogeme
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.results as res
from biogeme.expressions import Beta, DefineVariable, bioDraws, exp, log, MonteCarlo, Derive, PanelLikelihoodTrajectory
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold
#import biogeme.exceptions as excep
#from biogeme.expressions import *
import biogeme.draws as draws
# Perform Likely Ratio Test
#from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import norm
import math
import seaborn as sn
from IPython.display import Math, HTML

#import dataset
pandas = pd.read_table("http://transp-or.epfl.ch/data/swissmetro.dat")

#rearrange dataset
#return from work = commuter
pandas.loc[:, ('PURPOSE')][(pandas['PURPOSE']==5)] = 1
#return from shopping = shopping
pandas.loc[:, ('PURPOSE')][(pandas['PURPOSE']==6)] = 2
#return from business = business
pandas.loc[:, ('PURPOSE')][(pandas['PURPOSE']==7)] = 3
#return from leisure = leisure
pandas.loc[:, ('PURPOSE')][(pandas['PURPOSE']==8)] = 4

pandas = pandas[(pandas['AGE'] !=6 ) & \
                #(pandas['INCOME'] !=4 ) &\
                (pandas['PURPOSE'] !=9 ) &\
                (pandas['CHOICE'] !=0 )]


#define dictionaries for visualization  of variables names
age_dic = {1: 'age≤24', 2: '24<age≤39', 3: '39<age≤54', 4: '54<age≤65', 5: '65<age', 6: 'not known'}
purpose_dic = {1: 'Commuter', 2: 'Shopping', 3: 'Business', 4: 'Leisure', 5: 'Return from work', 6: 'Return from shopping', 7: 'Return from business', 8: 'Return from leisure', 9: 'other'}
choice_dic = {1: 'Train', 2: 'SM', 3: 'Car'}
pandas.loc[:, ('INCOME')][(pandas['INCOME']==0)] = 1
income_dic = {1: 'under 50', 2: 'between 50 and 100', 3: 'over 100', 4: 'unknown'}
dic = [age_dic, income_dic, purpose_dic, choice_dic]
ticket_dic = {0: 'None', 1: 'Two way with half price card', 2: 'One way with half price card', 3: 'Two way normal price', 4: 'One way normal price', 5: 'Half day', 6: 'Annual season ticket', 7: 'Annual season ticket Junior or Senior', 8: 'Free travel after 7pm card', 9: 'Group ticket', 10: 'Other'}

#load data into  database
database = db.Database("swissmetro",pandas)
globals().update(database.variables)
database.panel('ID')

## Define the draw generator
myRandomNumberGenerators = {'MLHS':draws.getLatinHypercubeDraws}
database.setRandomNumberGenerators(myRandomNumberGenerators)


#define variables
#variables
CAR_AV_SP =  DefineVariable('CAR_AV_SP',CAR_AV  * (  SP   !=  0  ),database)
TRAIN_AV_SP =  DefineVariable('TRAIN_AV_SP',TRAIN_AV  * (  SP   !=  0  ),database)

SM_COST    =  SM_CO   * (  GA   ==  0  )   
TRAIN_COST =  TRAIN_CO   * (  GA   ==  0  )

###Parameters to be estimated (Note not all parameters are used in all models!)
##Attributes
#Alternative specific constants
ASC_CAR        = Beta('ASC_CAR',0,None,None,1)
ASC_TRAIN      = Beta('ASC_TRAIN',0,None,None,0)
ASC_SM         = Beta('ASC_SM',0,None,None,0)

#Cost (Note: Assumed generic)
B_COST          = Beta('B_COST',0,None,None,0)
B_COST_BUSINESS = Beta('B_COST_BUSINESS',0,None,None,0)
B_COST_PRIVATE  = Beta('B_COST_PRIVATE',0,None,None,0)

#Time
B_TIME                = Beta('B_TIME',0,None,None,0)
B_TIME_CAR            = Beta('B_TIME_CAR',0,None,None,0)
B_TIME_TRAIN          = Beta('B_TIME_TRAIN',0,None,None,0)
B_TIME_SM             = Beta('B_TIME_SM',0,None,None,0)
B_TIME_PUB            = Beta('B_TIME_PUB',0,None,None,0)
B_TIME_CAR_BUSINESS   = Beta('B_TIME_CAR_BUSINESS',0,None,None,0)
B_TIME_TRAIN_BUSINESS = Beta('B_TIME_TRAIN_BUSINESS',0,None,None,0)
B_TIME_SM_BUSINESS    = Beta('B_TIME_SM_BUSINESS',0,None,None,0)
B_TIME_PUB_BUSINESS   = Beta('B_TIME_PUB_BUSINESS',0,None,None,0)
B_TIME_CAR_PRIVATE    = Beta('B_TIME_CAR_PRIVATE',0,None,None,0)
B_TIME_TRAIN_PRIVATE  = Beta('B_TIME_TRAIN_PRIVATE',0,None,None,0)
B_TIME_SM_PRIVATE     = Beta('B_TIME_SM_PRIVATE',0,None,None,0)
B_TIME_PUB_PRIVATE    = Beta('B_TIME_PUB_PRIVATE',0,None,None,0)

#HE (Note: Not available for car)
B_HE                = Beta('B_HE',0,None,None,0)
B_HE_TRAIN          = Beta('B_HE_TRAIN',0,None,None,0)
B_HE_SM             = Beta('B_HE_SM',0,None,None,0)
B_HE_BUSINESS       = Beta('B_HE_BUSINESS',0,None,None,0)
B_HE_TRAIN_BUSINESS = Beta('B_HE_TRAIN_BUSINESS',0,None,None,0)
B_HE_SM_BUSINESS    = Beta('B_HE_SM_BUSINESS',0,None,None,0)
B_HE_PRIVATE        = Beta('B_HE_PRIVATE',0,None,None,0)
B_HE_TRAIN_PRIVATE  = Beta('B_HE_TRAIN_PRIVATE',0,None,None,0)
B_HE_SM_PRIVATE     = Beta('B_HE_SM_PRIVATE',0,None,None,0)

#Seats (Note: Only avaliable for SM)
B_SEATS = Beta('B_SEATS',0,None,None,0)

##Characteristics
#Age
B_AGE_1_TRAIN      = Beta('B_AGE_1_TRAIN',0,None,None,1) #Note: Reference
B_AGE_2_TRAIN      = Beta('B_AGE_2_TRAIN',0,None,None,0)
B_AGE_3_TRAIN      = Beta('B_AGE_3_TRAIN',0,None,None,0)
B_AGE_4_TRAIN      = Beta('B_AGE_4_TRAIN',0,None,None,0)
B_AGE_5_TRAIN      = Beta('B_AGE_5_TRAIN',0,None,None,0)
B_AGE_6_TRAIN      = Beta('B_AGE_6_TRAIN',0,None,None,0)
B_AGE_1_SM         = Beta('B_AGE_1_SM',0,None,None,1) #Note: Reference
B_AGE_2_SM         = Beta('B_AGE_2_SM',0,None,None,0)
B_AGE_3_SM         = Beta('B_AGE_3_SM',0,None,None,0)
B_AGE_4_SM         = Beta('B_AGE_4_SM',0,None,None,0)
B_AGE_5_SM         = Beta('B_AGE_5_SM',0,None,None,0)
B_AGE_6_SM         = Beta('B_AGE_6_SM',0,None,None,0)
B_AGE_1_PUB        = Beta('B_AGE_1_PUB',0,None,None,1) #Note: Reference
B_AGE_2_PUB        = Beta('B_AGE_2_PUB',0,None,None,0)
B_AGE_3_PUB        = Beta('B_AGE_3_PUB',0,None,None,0)
B_AGE_4_PUB        = Beta('B_AGE_4_PUB',0,None,None,0)
B_AGE_5_PUB        = Beta('B_AGE_5_PUB',0,None,None,0)
B_AGE_6_PUB        = Beta('B_AGE_6_PUB',0,None,None,0)
B_AGE_ADULTS_TRAIN = Beta('B_AGE_TRAIN_ADULTS',0,None,None,0)
B_AGE_ADULTS_SM    = Beta('B_AGE_ADULTS_SM',0,None,None,0)
B_AGE_ADULTS_PUB   = Beta('B_AGE_ADULTS_PUB',0,None,None,0)

#Luggage
B_LUGGAGE_TRAIN = Beta('B_LUGGAGE_TRAIN', 0, None, None, 0)
B_LUGGAGE_SM    = Beta('B_LUGGAGE_SM', 0, None, None, 0)
B_LUGGAGE_PUB   = Beta('B_LUGGAGE_PUB', 0, None, None, 0)

#Gender
B_MALE_TRAIN = Beta('B_MALE_TRAIN',0,None,None,0)
B_MALE_SM    = Beta('B_MALE_SM',0,None,None,0)
B_MALE_PUB   = Beta('B_MALE_PUB',0,None,None,0)

#Purpose
B_BUSINESS       = Beta('B_BUSINESS',0,None,None,0)
B_BUSINESS_TRAIN = Beta('B_BUSINESS_TRAIN',0,None,None,0)
B_BUSINESS_SM    = Beta('B_BUSINESS_SM',0,None,None,0)
B_PRIVATE        = Beta('B_PRIVATE',0,None,None,0)
B_PRIVATE_TRAIN  = Beta('B_PRIVATE_TRAIN',0,None,None,0)
B_PRIVATE_SM     = Beta('B_PRIVATE_SM',0,None,None,0)
B_COMMUTER       = Beta('B_COMMUTER',0,None,None,0)
B_COMMUTER_TRAIN = Beta('B_COMMUTER_TRAIN',0,None,None,0)
B_COMMUTER_SM    = Beta('B_COMMUTER_SM',0,None,None,0)

#GA
B_GA        = Beta('B_GA',0,None,None,0)
B_GA_TRAIN  = Beta('B_GA_TRAIN',0,None,None,0)
B_GA_SM     = Beta('B_GA_SM',0,None,None,0)

#First
B_FIRST_TRAIN = Beta('B_FIRST_TRAIN',0,None,None,0)
B_FIRST_SM    = Beta('B_FIRST_SM',0,None,None,0)
B_FIRST       = Beta('B_FIRST',0,None,None,0)

##Non linearization
#Cost
q_COST = Beta('q_COST',1,None,None,0)

#Time
q_TIME       = Beta('q_TIME',1,None,None,0)
q_TIME_TRAIN = Beta('q_TIME_TRAIN',1,None,None,0)
q_TIME_SM    = Beta('q_TIME_SM',1,None,None,0)
q_TIME_CAR   = Beta('q_TIME_CAR',1,None,None,0)
q_TIME_PUB   = Beta('q_TIME_PUB',1,None,None,0)

#HE
q_HE = Beta('q_HE',1,None,None,0)

##Nesting parameter
MU = Beta('MU',1,0,1,0)

##ML RANDOM GENERIC TIME LOGNORMAL
BETA_TIME_mean = Beta('BETA_TIME_mean',0,None,None,0)
BETA_TIME_std = Beta('BETA_TIME_std',1,None,None,0)
BETA_TIME_random = -exp(BETA_TIME_mean + BETA_TIME_std * bioDraws('BETA_TIME_random','NORMAL'))

##ML RANDOM SPECIFIC TIME TRAIN LOGNORMAL
BETA_TIME_TRAIN_mean = Beta('BETA_TIME_TRAIN_mean',0,None,None,0)
BETA_TIME_TRAIN_std = Beta('BETA_TIME_TRAIN_std',1,None,None,0)
BETA_TIME_TRAIN_random = -exp(BETA_TIME_TRAIN_mean + BETA_TIME_TRAIN_std * bioDraws('BETA_TIME_TRAIN_random','NORMAL'))

##ML RANDOM SPECIFIC TIME SM  LOGNORMAL
BETA_TIME_SM_mean = Beta('BETA_TIME_SM_mean',0,None,None,0)
BETA_TIME_SM_std = Beta('BETA_TIME_SM_std',1,None,None,0)
BETA_TIME_SM_random = -exp(BETA_TIME_SM_mean + BETA_TIME_SM_std * bioDraws('BETA_TIME_SM_random','NORMAL'))

##ML RANDOM SPECIFIC TIME CAR LOGNORMAL
BETA_TIME_CAR_mean = Beta('BETA_TIME_CAR_mean',0,None,None,0)
BETA_TIME_CAR_std = Beta('BETA_TIME_CAR_std',1,None,None,0)
BETA_TIME_CAR_random = -exp(BETA_TIME_CAR_mean + BETA_TIME_CAR_std * bioDraws('BETA_TIME_CAR_random','NORMAL'))

##ML RANDOM GENERIC COST LOGNORMAL
BETA_COST_mean = Beta('BETA_COST_mean',0,None,None,0)
BETA_COST_std = Beta('BETA_COST_std',1,None,None,0)
BETA_COST_random = -exp(BETA_COST_mean + BETA_COST_std * bioDraws('BETA_COST_random','NORMAL'))

##ML RANDOM GENERIC HE LOGNORMAL
BETA_HE_mean = Beta('BETA_HE_mean',0,None,None,0)
BETA_HE_std = Beta('BETA_HE_std',1,None,None,0)
BETA_HE_random = -exp(BETA_HE_mean + BETA_HE_std * bioDraws('BETA_HE_random','NORMAL'))

##ML RANDOM GENERIC TIME NORMAL
BETA_TIME_mean_Norm = Beta('BETA_TIME_mean_Norm',0,None,None,0)
BETA_TIME_std_Norm = Beta('BETA_TIME_std_Norm',1,None,None,0)
BETA_TIME_random_Norm = BETA_TIME_mean_Norm + BETA_TIME_std_Norm * bioDraws('BETA_TIME_random_Norm','NORMAL')

##ML RANDOM SPECIFIC TIME TRAIN LOGNORMAL
BETA_TIME_TRAIN_mean_Norm = Beta('BETA_TIME_TRAIN_mean_Norm',0,None,None,0)
BETA_TIME_TRAIN_std_Norm = Beta('BETA_TIME_TRAIN_std_Norm',1,None,None,0)
BETA_TIME_TRAIN_random_Norm = BETA_TIME_TRAIN_mean_Norm + BETA_TIME_TRAIN_std_Norm * bioDraws('BETA_TIME_TRAIN_random_Norm','NORMAL')

##ML RANDOM SPECIFIC TIME SM  NORMAL
BETA_TIME_SM_mean_Norm = Beta('BETA_TIME_SM_mean_Norm',0,None,None,0)
BETA_TIME_SM_std_Norm = Beta('BETA_TIME_SM_std_Norm',1,None,None,0)
BETA_TIME_SM_random_Norm = BETA_TIME_SM_mean_Norm + BETA_TIME_SM_std_Norm * bioDraws('BETA_TIME_SM_random_Norm','NORMAL')

##ML RANDOM SPECIFIC TIME CAR NORMAL
BETA_TIME_CAR_mean_Norm = Beta('BETA_TIME_CAR_mean_Norm',0,None,None,0)
BETA_TIME_CAR_std_Norm = Beta('BETA_TIME_CAR_std_Norm',1,None,None,0)
BETA_TIME_CAR_random_Norm = BETA_TIME_CAR_mean_Norm + BETA_TIME_CAR_std_Norm * bioDraws('BETA_TIME_CAR_random_Norm','NORMAL')

##ML RANDOM GENERIC COST NORMAL
BETA_COST_mean_Norm = Beta('BETA_COST_mean_Norm',0,None,None,0)
BETA_COST_std_Norm = Beta('BETA_COST_std_Norm',1,None,None,0)
BETA_COST_random_Norm = BETA_COST_mean_Norm + BETA_COST_std_Norm * bioDraws('BETA_COST_random_Norm','NORMAL')

##ML RANDOM GENERIC HE NORMAL
BETA_HE_mean_Norm = Beta('BETA_HE_mean_Norm',0,None,None,0)
BETA_HE_std_Norm = Beta('BETA_HE_std_Norm',1,None,None,0)
BETA_HE_random_Norm = BETA_HE_mean_Norm + BETA_HE_std_Norm * bioDraws('BETA_HE_random_Norm','NORMAL')

##Scaling 'COST', 'TRAVEL-TIME' and 'HE' by a factor of 100 and adding the scaled variables to the database
 
TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED',\
                                 TRAIN_TT / 100.0,database)
TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED',\
                                   TRAIN_COST / 100,database)
TRAIN_HE_SCALED = DefineVariable('TRAIN_HE_SCALED',\
                                   TRAIN_HE / 100.0, database)

SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0,database)
SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100,database)
SM_HE_SCALED = DefineVariable('SM_HE_SCALED',\
                                   SM_HE / 100, database)

CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100,database)
CAR_COST_SCALED = DefineVariable('CAR_COST_SCALED', CAR_CO / 100,database)

###Defining new variables and adding columns to the database
#Age
AGE_1 = DefineVariable('AGE_1', (  AGE   ==  1  ),database) #don't scale because is cathegorical
AGE_2 = DefineVariable('AGE_2', (  AGE   ==  2  ),database)
AGE_3 = DefineVariable('AGE_3', (  AGE   ==  3  ),database)
AGE_4 = DefineVariable('AGE_4', (  AGE   ==  4  ),database)
AGE_5 = DefineVariable('AGE_5', (  AGE   ==  5  ),database)
AGE_6 = DefineVariable('AGE_6', (  AGE   ==  6  ),database)

#Purpose
PRIVATE  = DefineVariable("PRIVATE", (PURPOSE == 1), database)
COMMUTER = DefineVariable("COMMUTER", (PURPOSE == 2), database)
BUSINESS = DefineVariable("BUSINESS", (PURPOSE == 3), database)