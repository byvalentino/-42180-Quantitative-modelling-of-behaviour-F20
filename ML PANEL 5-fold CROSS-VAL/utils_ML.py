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

def likeRatioTest(restr_betas, unrestr_betas, restr_LL, unrestr_LL, name_unrestricted, name_restricted, prob = 0.95):
    '''
    Test for Nested Hypotheses
    restr_betas = number of parameters of the restricted model 
    unrestr_betas = number of parameters of the unrestricted model
    restr_LL = Loglikelihood of the restricted model
    unrestr_LL = Loglikelihood of the unrestricted model 
    prob = confidence of the Chi Squared Distribution
    dof = degrees of freedom for Chi Squared
    '''
    dof = unrestr_betas - restr_betas
    if dof == 0:
        print('NON nested models:LogLikelihood ratio test NOT applicable')
        return None
    
    stat = -2*(restr_LL-unrestr_LL)

    critical = chi2.ppf(prob, dof)
    
    print('RatioTest',abs(stat))
    print('Critical',critical)
        
    if abs(stat) >= critical:
        print('Chose Unrestricted (reject H0 the parameters of the unrestricted model are 0)')
        display(Markdown('### LL Ratio Test Result (abs(STAT)-CHI2)'+str(abs(stat) - critical)))
        return name_unrestricted
    else:
        print('Chose Restricted (fail to reject H0 the parameters of the unrestricted model are 0)')
        display(Markdown('### LL ratio test result '+str(abs(stat) - critical)))
        return name_restricted
    
# Perform Horowitz Test performed as BenAkiva-Swait

def BenAkivaSwaitHorowitzTest(unrestr_betas, restr_betas, N, J, rh0bs_U, rh0bs_R, name_unrestricted, name_restricted, threshold=0.07):
    '''
    Test for NON Nested Hypotheses
    Ben-Akiva p.171-172
    This test is for models that are not a restriction one of the other
    Probability that the model with greater ro0bs is false
    N = sample size
    J = number of choices
    rh0bs_U = rho bar squared unrestricted model
    rh0bs_R = rho bar squared restricted model
    unrestr_betas = number of parameters of the unrestricted model
    restr_betas = number of parameters of the restricted model
    '''
    if rh0bs_U - rh0bs_R >=0:
        n=1
    else:
        n=-1
    #print(n)    
    dof = n*(unrestr_betas - restr_betas)
    z = n*(rh0bs_U - rh0bs_R)
    
    try:
        Pr = norm.cdf(-math.sqrt(2*N*z*math.log(J) + dof), loc=0, scale=1)
        #print(Pr)
        prdiff = Pr-threshold
        if prdiff<0 and n==1:
            print('Probability that the unrestricted model is false having greater rho bar squere, is below threshold:')
            print('Threshold:'+str(threshold))
            print('Probability:'+str(Pr))
            print('Difference:'+str(Pr-threshold))
            return name_unrestricted
        if prdiff>=0 and n==1:
            print('Probability that the unrestricted model is false having greater rho bar squere, is above threshold:')
            print('Threshold:'+str(threshold))
            print('Probability:'+str(Pr))
            print('Difference:'+str(Pr-threshold))            
            return name_restricted
        if prdiff<0 and n==-1:
            print('Probability that the restricted model is false having greater rho bar squere, is below threshold:')
            print('Threshold:'+str(threshold))
            print('Probability:'+str(Pr))
            print('Difference:'+str(Pr-threshold))
            return name_restricted
        if prdiff>=0 and n==-1:
            print('Probability that the restricted model is false having greater rho bar squere, is above threshold:')
            print('Threshold:'+str(threshold))
            print('Probability:'+str(Pr))
            print('Difference:'+str(Pr-threshold))
            return name_unrestricted

    except:
    #Pr[(rho bar square U - rho bar square R)>0] <= Phi{-[2*N*z*ln(K)) - sqrt(unrestr_betas-restr_betas)]}
        print("math domain error")
        #print('degrees of freedom: '+str(dof))
        #print('rho bar square difference: '+str(z))
        display(Markdown('### degrees of freedom: '+str(dof)))
        display(Markdown('### rho bar square difference: '+str(z)))
        
        #print(Pr,z)

def print_result(results, database):
    '''
    print 
    Loglikelihood
    AIC
    BIC
    RHO bar square
    '''
    for r in results:
        display(Markdown(f"# Results model {r.data.modelName}"))
        print(r.getEstimatedParameters()[['Value','Std err','t-test','p-value']])
        print(f"LL(0) =    {r.data.initLogLike:.3f}")
        print(f"LL(beta) = {r.data.logLike:.3f}")
        print('***************')
        print(f"AIC = {2*r.getEstimatedParameters().shape[0] - 2*r.data.logLike}")
        print(f"BIC = {2*r.getEstimatedParameters().shape[0]*math.log(database.getNumberOfObservations()) - 2*r.data.logLike}")
        print(f"rho bar square = {r.data.rhoBarSquare:.3g}")
        #print(f"Output file: {results.data.htmlFileName}")
    if len(results) > 1:
        display(Markdown(f"### Diff. LL(beta) {results[-1].data.modelName} > {results[-2].data.modelName} <- {results[-1].data.logLike > results[-2].data.logLike}"))
        print('***************')
        print('***************')
        display(Markdown(f"### Diff. rho bar square {results[-1].data.modelName} > {results[-2].data.modelName} <- {results[-1].data.rhoBarSquare > results[-2].data.rhoBarSquare}"))
        print('***************')
        #print(f"Output file: {results[-1].data.htmlFileName}")   

def estimate_model(V
                   #, CHOICE, Car_SP, TRAIN_SP, SM_SP, CAR_AV_SP, TRAIN_AV_SP, SM_AV
                   , Draws
                   , ModName
                   , train
                   , myRandomNumberGenerators):
    db_train = db.Database("swissmetro_train",train)
    db_train.setRandomNumberGenerators(myRandomNumberGenerators)
    globals().update(db_train.variables)
    #locals().update(db_train.variables)
    db_train.panel('ID')
    
    #variables
    CAR_AV_SP =  DefineVariable('CAR_AV_SP',CAR_AV  * (  SP   !=  0  ),db_train)
    TRAIN_AV_SP =  DefineVariable('TRAIN_AV_SP',TRAIN_AV  * (  SP   !=  0  ),db_train)

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

    ##ML RANDOM GENERIC TIME LOGNORMAL
    BBETA_TIME_mean = Beta('BBETA_TIME_mean',0,None,None,0)
    BBETA_TIME_std = Beta('BBETA_TIME_std',1,None,None,0)
    BBETA_TIME_random = -exp(BBETA_TIME_mean + BBETA_TIME_std * bioDraws('BBETA_TIME_random','NORMAL'))

    ##ML RANDOM SPECIFIC TIME TRAIN LOGNORMAL
    BBETA_TIME_TRAIN_mean = Beta('BBETA_TIME_TRAIN_mean',0,None,None,0)
    BBETA_TIME_TRAIN_std = Beta('BBETA_TIME_TRAIN_std',1,None,None,0)
    BBETA_TIME_TRAIN_random = -exp(BBETA_TIME_TRAIN_mean + BBETA_TIME_TRAIN_std * bioDraws('BBETA_TIME_TRAIN_random','NORMAL'))

    ##ML RANDOM SPECIFIC TIME SM  LOGNORMAL
    BBETA_TIME_SM_mean = Beta('BBETA_TIME_SM_mean',0,None,None,0)
    BBETA_TIME_SM_std = Beta('BBETA_TIME_SM_std',1,None,None,0)
    BBETA_TIME_SM_random = -exp(BBETA_TIME_SM_mean + BBETA_TIME_SM_std * bioDraws('BBETA_TIME_SM_random','NORMAL'))

    ##ML RANDOM SPECIFIC TIME CAR LOGNORMAL
    BBETA_TIME_CAR_mean = Beta('BBETA_TIME_CAR_mean',0,None,None,0)
    BBETA_TIME_CAR_std = Beta('BBETA_TIME_CAR_std',1,None,None,0)
    BBETA_TIME_CAR_random = -exp(BBETA_TIME_CAR_mean + BBETA_TIME_CAR_std * bioDraws('BBETA_TIME_CAR_random','NORMAL'))

    ##ML RANDOM GENERIC COST LOGNORMAL
    BBETA_COST_mean = Beta('BBETA_COST_mean',0,None,None,0)
    BBETA_COST_std = Beta('BBETA_COST_std',1,None,None,0)
    BBETA_COST_random = -exp(BBETA_COST_mean + BBETA_COST_std * bioDraws('BBETA_COST_random','NORMAL'))

    ##ML RANDOM GENERIC HE LOGNORMAL
    BBETA_HE_mean = Beta('BBETA_HE_mean',0,None,None,0)
    BBETA_HE_std = Beta('BBETA_HE_std',1,None,None,0)
    BBETA_HE_random = -exp(BBETA_HE_mean + BBETA_HE_std * bioDraws('BBETA_HE_random','NORMAL'))

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

    ##ML RANDOM GENERIC TIME NORMAL
    BBETA_TIME_mean_Norm = Beta('BBETA_TIME_mean_Norm',0,None,None,0)
    BBETA_TIME_std_Norm = Beta('BBETA_TIME_std_Norm',1,None,None,0)
    BBETA_TIME_random_Norm = BBETA_TIME_mean_Norm + BBETA_TIME_std_Norm * bioDraws('BBETA_TIME_random_Norm','NORMAL')

    ##ML RANDOM SPECIFIC TIME TRAIN LOGNORMAL
    BBETA_TIME_TRAIN_mean_Norm = Beta('BBETA_TIME_TRAIN_mean_Norm',0,None,None,0)
    BBETA_TIME_TRAIN_std_Norm = Beta('BBETA_TIME_TRAIN_std_Norm',1,None,None,0)
    BBETA_TIME_TRAIN_random_Norm = BBETA_TIME_TRAIN_mean_Norm + BBETA_TIME_TRAIN_std_Norm * bioDraws('BBETA_TIME_TRAIN_random_Norm','NORMAL')

    ##ML RANDOM SPECIFIC TIME SM  NORMAL
    BBETA_TIME_SM_mean_Norm = Beta('BBETA_TIME_SM_mean_Norm',0,None,None,0)
    BBETA_TIME_SM_std_Norm = Beta('BBETA_TIME_SM_std_Norm',1,None,None,0)
    BBETA_TIME_SM_random_Norm = BBETA_TIME_SM_mean_Norm + BBETA_TIME_SM_std_Norm * bioDraws('BBETA_TIME_SM_random_Norm','NORMAL')

    ##ML RANDOM SPECIFIC TIME CAR NORMAL
    BBETA_TIME_CAR_mean_Norm = Beta('BBETA_TIME_CAR_mean_Norm',0,None,None,0)
    BBETA_TIME_CAR_std_Norm = Beta('BBETA_TIME_CAR_std_Norm',1,None,None,0)
    BBETA_TIME_CAR_random_Norm = BBETA_TIME_CAR_mean_Norm + BBETA_TIME_CAR_std_Norm * bioDraws('BBETA_TIME_CAR_random_Norm','NORMAL')

    ##ML RANDOM GENERIC COST NORMAL
    BBETA_COST_mean_Norm = Beta('BBETA_COST_mean_Norm',0,None,None,0)
    BBETA_COST_std_Norm = Beta('BBETA_COST_std_Norm',1,None,None,0)
    BBETA_COST_random_Norm = BBETA_COST_mean_Norm + BBETA_COST_std_Norm * bioDraws('BBETA_COST_random_Norm','NORMAL')

    ##ML RANDOM GENERIC HE NORMAL
    BBETA_HE_mean_Norm = Beta('BBETA_HE_mean_Norm',0,None,None,0)
    BBETA_HE_std_Norm = Beta('BBETA_HE_std_Norm',1,None,None,0)
    BBETA_HE_random_Norm = BBETA_HE_mean_Norm + BBETA_HE_std_Norm * bioDraws('BBETA_HE_random_Norm','NORMAL')

    ##Scaling 'COST', 'TRAVEL-TIME' and 'HE' by a factor of 100 and adding the scaled variables to the database

    TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED',\
                                     TRAIN_TT / 100.0,db_train)
    TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED',\
                                       TRAIN_COST / 100,db_train)
    TRAIN_HE_SCALED = DefineVariable('TRAIN_HE_SCALED',\
                                       TRAIN_HE / 100, db_train)

    SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0,db_train)
    SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100,db_train)
    SM_HE_SCALED = DefineVariable('SM_HE_SCALED',\
                                       SM_HE / 100, db_train)

    CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100,db_train)
    CAR_COST_SCALED = DefineVariable('CAR_COST_SCALED', CAR_CO / 100,db_train)

    ###Defining new variables and adding columns to the database
    #Age
    AGE_1 = DefineVariable('AGE_1', (  AGE   ==  1  ),db_train) #don't scale because is cathegorical
    AGE_2 = DefineVariable('AGE_2', (  AGE   ==  2  ),db_train)
    AGE_3 = DefineVariable('AGE_3', (  AGE   ==  3  ),db_train)
    AGE_4 = DefineVariable('AGE_4', (  AGE   ==  4  ),db_train)
    AGE_5 = DefineVariable('AGE_5', (  AGE   ==  5  ),db_train)
    AGE_6 = DefineVariable('AGE_6', (  AGE   ==  6  ),db_train)

    #Purpose
    PRIVATE  = DefineVariable("PRIVATE", (PURPOSE == 1), db_train)
    COMMUTER = DefineVariable("COMMUTER", (PURPOSE == 2), db_train)
    BUSINESS = DefineVariable("BUSINESS", (PURPOSE == 3), db_train)
    
    av = {3: CAR_AV_SP,1: TRAIN_AV_SP,2: SM_AV}
    obsprob = exp(models.loglogit(V,av,CHOICE))
    condprobIndiv = PanelLikelihoodTrajectory(obsprob)
    logprob = log(MonteCarlo(condprobIndiv))        
    bg = bio.BIOGEME(db_train,logprob,numberOfDraws=Draws)
    bg.modelName = ModName
    result = bg.estimate()
    return result

def test_model(V
               #, CHOICE, Car_SP, TRAIN_SP, SM_SP, CAR_AV_SP, TRAIN_AV_SP, SM_AV
               , R
               , Draws
               , ModName
               , test
               , myRandomNumberGenerators):
    
    db_test = db.Database("swissmetro_test",test)
    db_test.setRandomNumberGenerators(myRandomNumberGenerators)
    globals().update(db_test.variables)        
    #locals().update(db_test.variables)
    
    #define variables
    #variables
    CAR_AV_SP =  DefineVariable('CAR_AV_SP',CAR_AV  * (  SP   !=  0  ),db_test)
    TRAIN_AV_SP =  DefineVariable('TRAIN_AV_SP',TRAIN_AV  * (  SP   !=  0  ),db_test)

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

    ##ML RANDOM GENERIC TIME LOGNORMAL
    BBETA_TIME_mean = Beta('BBETA_TIME_mean',0,None,None,0)
    BBETA_TIME_std = Beta('BBETA_TIME_std',1,None,None,0)
    BBETA_TIME_random = -exp(BBETA_TIME_mean + BBETA_TIME_std * bioDraws('BBETA_TIME_random','NORMAL'))

    ##ML RANDOM SPECIFIC TIME TRAIN LOGNORMAL
    BBETA_TIME_TRAIN_mean = Beta('BBETA_TIME_TRAIN_mean',0,None,None,0)
    BBETA_TIME_TRAIN_std = Beta('BBETA_TIME_TRAIN_std',1,None,None,0)
    BBETA_TIME_TRAIN_random = -exp(BBETA_TIME_TRAIN_mean + BBETA_TIME_TRAIN_std * bioDraws('BBETA_TIME_TRAIN_random','NORMAL'))

    ##ML RANDOM SPECIFIC TIME SM  LOGNORMAL
    BBETA_TIME_SM_mean = Beta('BBETA_TIME_SM_mean',0,None,None,0)
    BBETA_TIME_SM_std = Beta('BBETA_TIME_SM_std',1,None,None,0)
    BBETA_TIME_SM_random = -exp(BBETA_TIME_SM_mean + BBETA_TIME_SM_std * bioDraws('BBETA_TIME_SM_random','NORMAL'))

    ##ML RANDOM SPECIFIC TIME CAR LOGNORMAL
    BBETA_TIME_CAR_mean = Beta('BBETA_TIME_CAR_mean',0,None,None,0)
    BBETA_TIME_CAR_std = Beta('BBETA_TIME_CAR_std',1,None,None,0)
    BBETA_TIME_CAR_random = -exp(BBETA_TIME_CAR_mean + BBETA_TIME_CAR_std * bioDraws('BBETA_TIME_CAR_random','NORMAL'))

    ##ML RANDOM GENERIC COST LOGNORMAL
    BBETA_COST_mean = Beta('BBETA_COST_mean',0,None,None,0)
    BBETA_COST_std = Beta('BBETA_COST_std',1,None,None,0)
    BBETA_COST_random = -exp(BBETA_COST_mean + BBETA_COST_std * bioDraws('BBETA_COST_random','NORMAL'))

    ##ML RANDOM GENERIC HE LOGNORMAL
    BBETA_HE_mean = Beta('BBETA_HE_mean',0,None,None,0)
    BBETA_HE_std = Beta('BBETA_HE_std',1,None,None,0)
    BBETA_HE_random = -exp(BBETA_HE_mean + BBETA_HE_std * bioDraws('BBETA_HE_random','NORMAL'))

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

    ##ML RANDOM GENERIC TIME NORMAL
    BBETA_TIME_mean_Norm = Beta('BBETA_TIME_mean_Norm',0,None,None,0)
    BBETA_TIME_std_Norm = Beta('BBETA_TIME_std_Norm',1,None,None,0)
    BBETA_TIME_random_Norm = BBETA_TIME_mean_Norm + BBETA_TIME_std_Norm * bioDraws('BBETA_TIME_random_Norm','NORMAL')

    ##ML RANDOM SPECIFIC TIME TRAIN LOGNORMAL
    BBETA_TIME_TRAIN_mean_Norm = Beta('BBETA_TIME_TRAIN_mean_Norm',0,None,None,0)
    BBETA_TIME_TRAIN_std_Norm = Beta('BBETA_TIME_TRAIN_std_Norm',1,None,None,0)
    BBETA_TIME_TRAIN_random_Norm = BBETA_TIME_TRAIN_mean_Norm + BBETA_TIME_TRAIN_std_Norm * bioDraws('BBETA_TIME_TRAIN_random_Norm','NORMAL')

    ##ML RANDOM SPECIFIC TIME SM  NORMAL
    BBETA_TIME_SM_mean_Norm = Beta('BBETA_TIME_SM_mean_Norm',0,None,None,0)
    BBETA_TIME_SM_std_Norm = Beta('BBETA_TIME_SM_std_Norm',1,None,None,0)
    BBETA_TIME_SM_random_Norm = BBETA_TIME_SM_mean_Norm + BBETA_TIME_SM_std_Norm * bioDraws('BBETA_TIME_SM_random_Norm','NORMAL')

    ##ML RANDOM SPECIFIC TIME CAR NORMAL
    BBETA_TIME_CAR_mean_Norm = Beta('BBETA_TIME_CAR_mean_Norm',0,None,None,0)
    BBETA_TIME_CAR_std_Norm = Beta('BBETA_TIME_CAR_std_Norm',1,None,None,0)
    BBETA_TIME_CAR_random_Norm = BBETA_TIME_CAR_mean_Norm + BBETA_TIME_CAR_std_Norm * bioDraws('BBETA_TIME_CAR_random_Norm','NORMAL')

    ##ML RANDOM GENERIC COST NORMAL
    BBETA_COST_mean_Norm = Beta('BBETA_COST_mean_Norm',0,None,None,0)
    BBETA_COST_std_Norm = Beta('BBETA_COST_std_Norm',1,None,None,0)
    BBETA_COST_random_Norm = BBETA_COST_mean_Norm + BBETA_COST_std_Norm * bioDraws('BBETA_COST_random_Norm','NORMAL')

    ##ML RANDOM GENERIC HE NORMAL
    BBETA_HE_mean_Norm = Beta('BBETA_HE_mean_Norm',0,None,None,0)
    BBETA_HE_std_Norm = Beta('BBETA_HE_std_Norm',1,None,None,0)
    BBETA_HE_random_Norm = BBETA_HE_mean_Norm + BBETA_HE_std_Norm * bioDraws('BBETA_HE_random_Norm','NORMAL')

    ##Scaling 'COST', 'TRAVEL-TIME' and 'HE' by a factor of 100 and adding the scaled variables to the database

    TRAIN_TT_SCALED = DefineVariable('TRAIN_TT_SCALED',\
                                     TRAIN_TT / 100.0,db_test)
    TRAIN_COST_SCALED = DefineVariable('TRAIN_COST_SCALED',\
                                       TRAIN_COST / 100,db_test)
    TRAIN_HE_SCALED = DefineVariable('TRAIN_HE_SCALED',\
                                       TRAIN_HE / 100, db_test)

    SM_TT_SCALED = DefineVariable('SM_TT_SCALED', SM_TT / 100.0,db_test)
    SM_COST_SCALED = DefineVariable('SM_COST_SCALED', SM_COST / 100,db_test)
    SM_HE_SCALED = DefineVariable('SM_HE_SCALED',\
                                       SM_HE / 100, db_test)

    CAR_TT_SCALED = DefineVariable('CAR_TT_SCALED', CAR_TT / 100,db_test)
    CAR_COST_SCALED = DefineVariable('CAR_COST_SCALED', CAR_CO / 100,db_test)

    ###Defining new variables and adding columns to the database
    #Age
    AGE_1 = DefineVariable('AGE_1', (  AGE   ==  1  ),db_test) #don't scale because is cathegorical
    AGE_2 = DefineVariable('AGE_2', (  AGE   ==  2  ),db_test)
    AGE_3 = DefineVariable('AGE_3', (  AGE   ==  3  ),db_test)
    AGE_4 = DefineVariable('AGE_4', (  AGE   ==  4  ),db_test)
    AGE_5 = DefineVariable('AGE_5', (  AGE   ==  5  ),db_test)
    AGE_6 = DefineVariable('AGE_6', (  AGE   ==  6  ),db_test)

    #Purpose
    PRIVATE  = DefineVariable("PRIVATE", (PURPOSE == 1), db_test)
    COMMUTER = DefineVariable("COMMUTER", (PURPOSE == 2), db_test)
    BUSINESS = DefineVariable("BUSINESS", (PURPOSE == 3), db_test)
    
    av = {3: CAR_AV_SP,1: TRAIN_AV_SP,2: SM_AV}
    prob1 = biogeme.expressions.MonteCarlo(exp(models.loglogit(V,av,1)))
    prob2 = biogeme.expressions.MonteCarlo(exp(models.loglogit(V,av,2)))
    prob3 = biogeme.expressions.MonteCarlo(exp(models.loglogit(V,av,3)))
    simulate = {'Prob. TRAIN': prob1,
                'Prob. SM': prob2,
                'Prob. CAR': prob3}
    #print(simulate)
    biosim = bio.BIOGEME(db_test, simulate, numberOfDraws=Draws)
    biosim.modelName = ModName  
    SR = biosim.simulate(R.data.betaValues)        
    
    return SR

def train_test_split_wrap(pandas, kfold=5, shuffle=True, random_state=17):
    #kf = KFold(n_splits = 3, shuffle = True)
    #splits = kf.split(pandas)
    train_idx = []
    test_idx = []
    skf = StratifiedKFold(n_splits=kfold, shuffle=shuffle, random_state=random_state)
    splits = skf.split(pandas, pandas['PURPOSE'])  
    for index in splits:
        train_idx.append(index[0])
        test_idx.append(index[1])
    return train_idx, test_idx


def cross_validation(names_list
                     , pandas
                     #, CHOICE, Car_SP, TRAIN_SP, SM_SP, CAR_AV_SP, TRAIN_AV_SP, SM_AV
                     , myRandomNumberGenerators
                     , ModNameSeqence=[]
                     , vSeq=[]
                     , Draws=100
                     , train_idx=None
                     , test_idx=None):
    #print(names_list)
    names_idx = [ ModNameSeqence.index(n) for n in names_list]  
    #print(names_idx)
    display(Markdown(f"# Perform  Cross Validation Between {list(names_list)}"))

    LLL = [0 for i in range(len(names_list))]
    LLLstd = [0 for i in range(len(names_list))]
    R = [[] for i in range(len(names_list))]
    LL= [[] for i in range(len(names_list))]
    
    if train_idx == None or test_idx  == None: 
        train_idx, test_idx = train_test_split_wrap(pandas)
    
    for i,m in enumerate(names_idx):
        for jj, index in enumerate(zip(train_idx,test_idx)):
            train = pandas.iloc[index[0]].copy()
            test = pandas.iloc[index[1]].copy()    
            R[i].append(estimate_model(vSeq[m]
                                       #, CHOICE, Car_SP, TRAIN_SP, SM_SP, CAR_AV_SP, TRAIN_AV_SP, SM_AV
                                       , Draws
                                       , ModNameSeqence[m]+'_tr_'+str(i)+'_loop_'+str(jj)
                                       , train
                                       , myRandomNumberGenerators))
            SR = test_model(vSeq[m]
                            #, CHOICE, Car_SP, TRAIN_SP, SM_SP, CAR_AV_SP, TRAIN_AV_SP, SM_AV
                            , R[i][jj]
                            , Draws
                            , ModNameSeqence[m]+'_te_'+str(i)+'_loop_'+str(jj)
                            , test
                            , myRandomNumberGenerators)
            loglike = 0
            #print(test.CHOICE.values.tolist())
            for ii,c in enumerate(test.CHOICE.values.tolist()):
                loglike += math.log(SR.iloc[ii,c-1]) #e.g. column 0 corresponds to choice 1
            #print(loglike)
            LL[i].append(loglike)
            print('Model',i)
            print('Fold',jj)
            #print(i,jj,sum(LL[i]))
            #LLL[i]=sum(LL[i])/len(LL[i])
            LLL[i]=np.mean(LL[i])
            LLLstd[i]=np.std(LL[i])
            #print(f'best model after {jj+1} loops: {names_list[LLL.index(max(LLL))]}')
    idx_Max = LLL.index(max(LLL))
    idx_min = LLL.index(min(LLL))
    best = names_list[idx_Max]
    print(f'loglikelihood model {names_list[idx_Max]}: {LLL[idx_Max]} +/- {2*LLLstd[idx_Max]}')
    print(f'loglikelihood model {names_list[idx_min]}: {LLL[idx_min]} +/- {2*LLLstd[idx_min]}')
    display(Markdown(f'## Accept {best} performs better off the sample'))
    return best, LLL

def estimate_N_compare(name_unrestricted
                       , name_restricted
                       , vSeq
                       , av
                       , CHOICE, Car_SP, TRAIN_SP, SM_SP, CAR_AV_SP, TRAIN_AV_SP, SM_AV
                       , database
                       , pandas
                       , myRandomNumberGenerators
                       , results
                       , ModNameSeqence=[]
                       , Draws=[]
                       , J=3
                       , Prob=0.95
                       , threshold=0.07
                       , testOnly=False
                       , panelEst=True):
    '''
    Estimate one model and compare it with the other
    2 tests are applied
    Loglikelihood
    BenakivaHorowitzTest
    '''
    
    u_ix = ModNameSeqence.index(name_unrestricted)
    r_ix = ModNameSeqence.index(name_restricted)
    
    
    

    if not testOnly:
        display(Markdown(f'## Estimate  {ModNameSeqence[u_ix]} and compare with {ModNameSeqence[r_ix]}'))

        if panelEst:
            #database.panel('ID')
            # Conditional to B_TIME_RND, the likelihood of one observation is
            # given by the logit model (called the kernel)
            obsprob = exp(models.loglogit(vSeq[u_ix],av,CHOICE))

            # Conditional to B_TIME_RND, the likelihood of all observations for
            # one individual (the trajectory) is the product of the likelihood of
            # each observation.
            condprobIndiv = PanelLikelihoodTrajectory(obsprob)

            # We integrate over B_TIME_RND using Monte-Carlo
            logprob = log(MonteCarlo(condprobIndiv))
        else:
            prob = exp(models.loglogit(vSeq[u_ix],av,CHOICE))
            logprob = log(MonteCarlo(prob))


        # Create the Biogeme object
        biogeme = bio.BIOGEME(database,logprob,numberOfDraws=Draws)
        biogeme.modelName = ModNameSeqence[u_ix]

        # Estimate the parameters
        results.append(biogeme.estimate())
        
    else:
        print('Compare',ModNameSeqence[u_ix],' with',ModNameSeqence[r_ix])
    #TEST LOGLIKERATIO
    restr_betas = results[r_ix].getEstimatedParameters().shape[0]
    unrestr_betas =  results[u_ix].getEstimatedParameters().shape[0]
    
    if restr_betas > unrestr_betas:
        k = u_ix
        u_ix = r_ix
        r_ix = k
        restr_betas = results[r_ix].getEstimatedParameters().shape[0]
        unrestr_betas =  results[u_ix].getEstimatedParameters().shape[0]
    
    display(Markdown(f'#### Unrestricted model {ModNameSeqence[u_ix]} and Restricted model {ModNameSeqence[r_ix]}'))
    print_result([results[u_ix],results[r_ix]], database)
    restr_LL = results[r_ix].data.logLike
    unrestr_LL = results[u_ix].data.logLike
    
    LikeRes = likeRatioTest(restr_betas, unrestr_betas, restr_LL, unrestr_LL, name_unrestricted, name_restricted, prob = Prob)
    
    #TEST NON NESTED
    N=database.getNumberOfObservations()
    rh0bs_U = results[u_ix].data.rhoBarSquare
    rh0bs_R = results[r_ix].data.rhoBarSquare
    BenRes = BenAkivaSwaitHorowitzTest(unrestr_betas, restr_betas, N, J, rh0bs_U, rh0bs_R, name_unrestricted, name_restricted, threshold=threshold)
    
    display(Markdown(f'#### LogLikelihood {LikeRes} - BenAkivaSwaitHorowitzTest {BenRes}'))
    if BenRes == LikeRes:
        return LikeRes, results
    else:
        display(Markdown(f'#### LogLikelihood TEST {LikeRes} disagrees with BenAkivaSwaitHorowitzTest {BenRes}'))
        display(Markdown(f'### TRY  CROSS VALIDATION'))
        
        output = cross_validation([ModNameSeqence[u_ix],ModNameSeqence[r_ix]]
                                  , pandas
                                  #, CHOICE, Car_SP, TRAIN_SP, SM_SP, CAR_AV_SP, TRAIN_AV_SP, SM_AV
                                  , myRandomNumberGenerators
                                  , ModNameSeqence=ModNameSeqence
                                  , vSeq=vSeq
                                  , Draws=100)
        return output[0], results

'''
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
'''