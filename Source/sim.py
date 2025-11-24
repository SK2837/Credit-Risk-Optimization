'''
Sim class is responsible for the generation of loan applications and their 
characteristics. Simulation parameters are substituted with ''s and 'np.nan's 
for confidentiality reasons.
'''

# import external packages
import numpy as np
import pandas as pd
import gc

class Sim:
    # initialize simulation parameters
    def __init__(self, distortions = {'e': 1, 'news_positives_score_bias': 0, 'repeats_positives_score_bias': 0, 'news_negatives_score_bias': 0, 'repeats_negatives_score_bias': 0, 'news_default_rate_bias': 0, 'repeats_default_rate_bias': 0, 'late_payment_rate_bias': 0, 'ar_effect': 0}):
        
        # distortion parameters
        self.distortions = distortions
        self.e = distortions['e']  # noize parameter
        self.news_positives_score_bias = distortions['news_positives_score_bias'] # factor increase in mean positives scores for new clients
        self.repeats_positives_score_bias = distortions['repeats_positives_score_bias'] # factor increase in mean positives scores for repeat clients
        self.news_negatives_score_bias = distortions['news_negatives_score_bias'] # factor decrease in mean negatives scores for new clients
        self.repeats_negatives_score_bias = distortions['repeats_negatives_score_bias'] # factor decrease in mean negatives scores for repeat clients
        self.news_default_rate_bias = distortions['news_default_rate_bias'] # factor change in segment default rates for new clients
        self.repeats_default_rate_bias = distortions['repeats_default_rate_bias'] # factor change in segment default rates for repeat clients
        self.late_payment_rate_bias = distortions['late_payment_rate_bias'] # factor change in segment late payment rates
        self.ar_effect = distortions['ar_effect'] # coefficient for acceptance rate dependence for applications number
        
        # constants
        self.ar_historical = 0.5 # historical acceptance rate
        self.c_ar_new = 0.1 # acceptance rate effect coefficient for new clients
        self.c_ar_repeat = 0.1 # acceptance rate effect coefficient for repeat clients
        
        # credit scoring model performance
        self.new_negative_score_mean = 40 # score mean for new good applications
        self.new_negative_score_std = 10 # score std for new good applications
        self.new_positive_score_mean = 60 # score mean for new bad applications
        self.new_positive_score_std = 10 # score std for new bad applications
        self.repeat_negative_score_mean = 45 # score mean for repeat good applications
        self.repeat_negative_score_std = 10 # score std for repeat good applications
        self.repeat_positive_score_mean = 65 # score mean for repeat bad applications
        self.repeat_positive_score_std = 10 # score std for repeat bad applications
        
        # distort model performance
        new_score_dif = self.new_negative_score_mean - self.new_positive_score_mean
        repeat_score_dif = self.repeat_negative_score_mean - self.repeat_positive_score_mean
        
        # adjust model performance according to assumptions
        self.new_positive_score_mean += self.news_positives_score_bias * new_score_dif
        self.repeat_positive_score_mean += self.repeats_positives_score_bias * repeat_score_dif
        self.new_negative_score_mean -= self.news_negatives_score_bias * new_score_dif
        self.repeat_negative_score_mean -= self.repeats_negatives_score_bias * repeat_score_dif
        
        # loan application segment estimates
        # 0 - customer segment, 
        # 1 - frequency among all applications, 
        # 2 - loan sum proportion among all applications, 
        # 3 - probability of going 60 days overdue, 
        # 4 - average score with current credit scoring model, 
        # 5 - average loan sum, 
        # 6 - average loan duration, 
        # 7 - average number of loans, 
        # 8 - probability of paying when overdue, 
        # 9 - average profit value
        
        self.new_loans = {
        1:['',0.2,0.2,0.1,60,1000,30,1,0.5,100],
        2:['',0.2,0.2,0.1,60,1000,30,1,0.5,100],
        3:['',0.2,0.2,0.1,60,1000,30,1,0.5,100],
        4:['',0.2,0.2,0.1,60,1000,30,1,0.5,100],
        5:['',0.1,0.1,0.1,60,1000,30,1,0.5,100],
        6:['',0.1,0.1,0.1,60,1000,30,1,0.5,100],
        }
        
        self.repeat_loans = {
        7:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        8:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        9:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        10:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        11:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        12:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        13:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        14:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        15:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        16:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        17:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        18:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        19:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        20:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        21:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        22:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        23:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        24:['',0.05,0.05,0.05,65,1500,45,2,0.6,150],
        25:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        26:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        27:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        28:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        29:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        30:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        31:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        32:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        33:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        34:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        35:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        36:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        37:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        38:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        39:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        40:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        41:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        42:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        43:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        44:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        45:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        46:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        47:['',0.02,0.02,0.05,65,1500,45,2,0.6,150],
        48:['',0.02,0.02,0.05,65,1500,45,2,0.6,150]
        }
            
        
        # distort segment estimates
        for x in self.new_loans:
            self.new_loans[x][3] *= 1 + self.news_default_rate_bias
            self.new_loans[x][3] = self.new_loans[x][3] if self.new_loans[x][3] <= 1 else 1
            self.new_loans[x][8] *= 1 + self.late_payment_rate_bias
            self.new_loans[x][8] = self.new_loans[x][8] if self.new_loans[x][8] <= 1 else 1
            
        for x in self.repeat_loans:
            self.repeat_loans[x][3] *= 1 + self.repeats_default_rate_bias
            self.repeat_loans[x][3] = self.repeat_loans[x][3] if self.repeat_loans[x][3] <= 1 else 1
            self.repeat_loans[x][8] *= 1 + self.late_payment_rate_bias
            self.repeat_loans[x][8] = self.repeat_loans[x][8] if self.repeat_loans[x][8] <= 1 else 1
            
        # debt segments
        self.debt_ranges = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700]
        self.debt_probabilities = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
        
        # accepted applications and accepted rate
        self.all_accepted = pd.DataFrame(data=[])
        self.ar = 0
    
    # generate dataframe of weekly loan applications
    def generateInput(self, iteration = 1):
        
        # received applications simulation parameters
        trend_change = 50 ### week of trend change
        
        # simulate number of weekly applications
        if (iteration <= trend_change):
            # data generating process before the trend change
            
            # generating the total number of weekly new applications
            no_of_weekly_new_applications = (10*(iteration)) - (0.1*((iteration)**2))
            no_of_weekly_new_applications = no_of_weekly_new_applications + np.random.normal(0,10) * self.e
            no_of_weekly_new_applications = no_of_weekly_new_applications if no_of_weekly_new_applications > 0 else 10
            # generating the total number of weekly repeat applications
            no_of_weekly_repeat_applications = (5*(iteration)) - (0.05*((iteration)**2)) - (0.001*((iteration)**3)) + (0.5*no_of_weekly_new_applications)
            no_of_weekly_repeat_applications = no_of_weekly_repeat_applications + np.random.normal(0,10) * self.e
            no_of_weekly_repeat_applications = no_of_weekly_repeat_applications if no_of_weekly_repeat_applications > 0 else 0
        else:                                                                                          
            # data generating process after the trend change
            
            # generating the total number of weekly new applications
            no_of_weekly_new_applications = 100 + (2*(iteration)) - (0.05*((iteration)**2))
            no_of_weekly_new_applications += (self.ar - self.ar_historical) * self.c_ar_new * (iteration - trend_change) * self.ar_effect
            no_of_weekly_new_applications += np.random.normal(0,10) * self.e
            no_of_weekly_new_applications = no_of_weekly_new_applications if no_of_weekly_new_applications > 0 else 10
            # generating the total number of weekly repeat applications
            no_of_weekly_repeat_applications = 50 + (1*(iteration)) - (0.02*((iteration)**2)) + (0.0005*((iteration)**3)) + (0.5*no_of_weekly_new_applications)
            no_of_weekly_repeat_applications += (self.ar - self.ar_historical) * self.c_ar_repeat * (iteration - trend_change) * self.ar_effect
            no_of_weekly_repeat_applications += np.random.normal(0,10) * self.e
        
        # scale volumes
        no_of_weekly_new_applications *= 1
        no_of_weekly_repeat_applications *= 1
        
        
        # generate application characteristics
        weekly_applications = pd.DataFrame(data=[])
        
        # generate new client loan application characteristics
        for i in range(1, (int(no_of_weekly_new_applications)+1)):
            probs = np.array([self.new_loans[x][1] for x in range(1, len(self.new_loans) + 1)])
            probs /= probs.sum()
            loantype = round((np.random.choice(np.arange(1, len(self.new_loans) + 1), p=probs))) # segment
            id = 'new_' + str(iteration) + '_' + str(i) # unique id
            sum = round(self.new_loans[loantype][5], 0) # loan sum
            duration = round((self.new_loans[loantype][6])/7, 0) # loan duration
            debt = self.debt_ranges[np.random.choice(np.arange(0, len(self.debt_probabilities)), p=self.debt_probabilities)] # outstanding debt
            try:
                dca_probability = (self.new_loans[loantype][3]) #+ ((0.0001*debt**2) - (0.001*debt)) # probability of going overdue
                dca = np.random.binomial(1, dca_probability) # if goes overdue
            except:
                dca = np.random.binomial(1, 0.1) # if the probability > 1
            late_payment = 1 if dca * np.random.binomial(1, self.new_loans[loantype][8]) == 1 else 0 # if repays after going overdue
            loan_value = self.new_loans[loantype][9] # profit value
            score = np.random.normal(self.new_negative_score_mean, self.new_negative_score_std) if dca == 0 else np.random.normal(self.new_positive_score_mean, self.new_positive_score_std) # credit score
            #score -= debt/17 # adjust for debt
            
            # store characteristics
            weekly_applications.loc[id, 'iteration'] = iteration
            weekly_applications.loc[id, 'maturation_at'] = iteration + duration
            weekly_applications.loc[id, 'repeat'] = False
            weekly_applications.loc[id, 'sum'] = int(sum)
            weekly_applications.loc[id, 'duration'] = int(round(self.new_loans[loantype][6], 0))
            weekly_applications.loc[id, 'debt'] = debt
            weekly_applications.loc[id, 'score'] = score
            weekly_applications.loc[id, 'dca'] = bool(dca)
            weekly_applications.loc[id, 'dca_at'] = iteration + duration + 10 if dca == 1 else 'NA'
            weekly_applications.loc[id, 'late_payment'] = late_payment
            weekly_applications.loc[id, 'late_payment_at'] = iteration + duration + int(np.random.uniform(1, 30)) if late_payment == 1 else 'NA'
            weekly_applications.loc[id, 'profit'] =  loan_value
        
        # generate repeat client loan application characteristics                           
        for i in range(1, (int(no_of_weekly_repeat_applications)+1)):
            probs = np.array([self.repeat_loans[x][1] for x in range(len(self.new_loans) + 1, len(self.new_loans) + len(self.repeat_loans) + 1)])
            probs /= probs.sum()
            loantype = round((np.random.choice(np.arange(len(self.new_loans) + 1, len(self.new_loans) + len(self.repeat_loans) + 1), p=probs))) # segment
            id = 'repeat_' + str(iteration) + '_' + str(i) # unique id
            sum = round(self.repeat_loans[loantype][5], 0) # loan sum
            duration = round((self.repeat_loans[loantype][6])/7, 0) # loan duration
            debt = self.debt_ranges[np.random.choice(np.arange(0, len(self.debt_probabilities)), p=self.debt_probabilities)] # outstanding debt
            try:
                dca_probability = self.repeat_loans[loantype][3] #+ ((0.0001*debt**2) - (0.001*debt)) # probability of going overdue
                dca = np.random.binomial(1, dca_probability) # if goes overdue
            except:
                dca = np.random.binomial(1, 0.1) # if the probability > 1
            late_payment = 1 if dca * np.random.binomial(1, self.repeat_loans[loantype][8]) == 1 else 0 # if repays after going overdue
            loan_value = self.repeat_loans[loantype][9] # profit value                                           
            score = np.random.normal(self.repeat_negative_score_mean, self.repeat_negative_score_std) if dca == 0 else np.random.normal(self.repeat_positive_score_mean, self.repeat_positive_score_std) # credit score
            #score -= debt/17 # adjust for debt
            
            # store characteristics
            weekly_applications.loc[id, 'iteration'] = iteration
            weekly_applications.loc[id, 'maturation_at'] = iteration + duration
            weekly_applications.loc[id, 'repeat'] = True
            weekly_applications.loc[id, 'sum'] = int(sum)
            weekly_applications.loc[id, 'duration'] = int(round(self.repeat_loans[loantype][6], 0))
            weekly_applications.loc[id, 'debt'] = debt
            weekly_applications.loc[id, 'score'] = score
            weekly_applications.loc[id, 'dca'] = bool(dca)
            weekly_applications.loc[id, 'dca_at'] = iteration + duration + 10 if dca == 1 else 'NA'
            weekly_applications.loc[id, 'late_payment'] = late_payment
            weekly_applications.loc[id, 'late_payment_at'] = iteration + duration + int(np.random.uniform(1, 30)) if late_payment == 1 else 'NA'
            weekly_applications.loc[id, 'profit'] =  loan_value                       
    
        return weekly_applications
    
    # performs the loan application acceptance decision
    def accept(self, app, threshold = 50):
        if app['score'] < threshold:
            app['accept'] = False
            return app
        else: 
            app['accept'] = True
            return app
    
    # generates dataframe of loan applications and ids of paid, overdue and paid after overdue loans for current week
    def simulate(self, i, weekly_applications, threshold = 50):
        
        weekly_applications = weekly_applications.apply(self.accept, axis = 1, args = [threshold])
        
        if 'accept' in weekly_applications.columns:
            accepted = weekly_applications.loc[weekly_applications['accept'] == True]
            self.ar = weekly_applications['accept'].mean()
            self.all_accepted = pd.concat([self.all_accepted, accepted])
            
            del accepted
                
        if not self.all_accepted.empty:    
            matured = self.all_accepted.loc[self.all_accepted['maturation_at'] == i]
            matured = matured.index
            dca = self.all_accepted.loc[self.all_accepted['dca_at'] == i]
            dca = dca.index
            paid_dca = self.all_accepted.loc[self.all_accepted['late_payment_at'] == i]
            paid_dca = paid_dca.index
            paid = self.all_accepted.loc[(self.all_accepted['maturation_at'] == i) & (self.all_accepted['dca'] == False)]
            paid = paid.index
            
            del matured
            
        else:
            self.ar = 0
            matured, dca, paid_dca, paid = [], [], [], []
            
        output = weekly_applications#[['iteration', 'sum', 'duration', 'score', 'repeat', 'accept', 'dca', 'profit']]
        
        gc.collect()
        del weekly_applications
        
        return output, paid, dca, paid_dca