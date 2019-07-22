import json
import math
import sys
from sklearn.model_selection import KFold
import numpy as np
import time
import matplotlib.pyplot as plt

class Hmm(object):
    def __init__(self, model_name):
        self.model = json.loads(open(model_name).read())["hmm"]
        self.A = self.model["A"]
        self.states = list(self.A.keys()) 
        self.N = len(self.states) 
        self.B = self.model["B"]
        self.symbols = list(list(self.B.values())[0].keys()) 
        self.M = len(self.symbols)
        self.pi = self.model["pi"]
        self.logA = {}
        self.logB = {}
        self.logpi = {}
        self.log_likelihood()
        return

    def log_likelihood(self):        
        for y in self.states:
            self.logA[y] = {}
            for y1 in self.A[y].keys():
                    self.logA[y][y1] = math.log(self.A[y][y1])
            self.logB[y] = {}
            for sym in self.B[y].keys():
                if self.B[y][sym] == 0:
                    self.logB[y][sym] =  sys.float_info.min 
                else:
                    self.logB[y][sym] = math.log(self.B[y][sym])
            if self.pi[y] == 0:
                self.logpi[y] =  sys.float_info.min 
            else:
                self.logpi[y] = math.log(self.pi[y])                

    def backward(self, obs):
       
        self.bwk = [{} for t in range(len(obs))]
        self.bwk_scaled = [{} for t in range(len(obs))]
        
        T = len(obs)
        for y in self.states:
            self.bwk[T-1][y] = 1
            try:
                self.bwk_scaled[T-1][y] = self.clist[T-1] * 1.0 
            except:
                print("EXCEPTION OCCURED in backward_scaled, T -1 = ", T -1)
            
        for t in reversed(range(T-1)):
            beta_local = {}
            for y in self.states:
                beta_local[y] = sum((self.bwk_scaled[t+1][y1] * self.A[y][y1] * self.B[y1][obs[t+1]]) for y1 in self.states)
                
            for y in self.states:
                self.bwk_scaled[t][y] = self.clist[t] * beta_local[y]
        
        log_p = -sum([math.log(c) for c in self.clist])

        return log_p 

    def normalize(self, alpha, states):
        alpha_sum = 0.0
        for y in states:
            alpha_sum += alpha[y]
        if alpha_sum != 0:
            normalizer = 1.0 / alpha_sum
        return normalizer

    def forward(self, obs):
        self.fwd = [{}]
        local_alpha = {}
        self.clist = [] 
        self.fwd_scaled = [{}] 
        for y in self.states:
            self.fwd[0][y] = self.pi[y] * self.B[y][obs[0]]
        c1 = self.normalize(self.fwd[0], self.states)
        self.clist.append(c1)
        for y in self.states:
            self.fwd_scaled[0][y] = c1 * self.fwd[0][y]
        for t in range(1, len(obs)):
            self.fwd.append({})     
            self.fwd_scaled.append({})     
            for y in self.states:
                local_alpha[y] = sum((self.fwd_scaled[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)

            c1 = self.normalize(local_alpha, self.states)
            self.clist.append(c1)
            for y in self.states:
                self.fwd_scaled[t][y] = c1 * local_alpha[y]

        log_p = -sum([math.log(c) for c in self.clist])      
        return log_p

    def viterbi(self, obs):
        vit = [{}]
        path = {}     
        for y in self.states:
            vit[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]
     

        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}     
            for y in self.states:
                (prob, state) = max((vit[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]     

            path = newpath
        n = 0           
        if len(obs)!=1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])
        
    def compute_aij(self, tables, i, j):
        zi_table = tables["zi_table"] 
        gamma_table = tables["gamma_table"] 
        numerator = 0.0
        denominator = 0.0
        
        for k in range(len(zi_table)):
            for t in range(len(zi_table[k]) - 1): 
                denominator += gamma_table[k][t][i] 
                numerator += zi_table[k][t][i][j] 
        aij = numerator / denominator
        return aij


    def compute_bj(self, tables, i, obslist, symbol):
        threshold = 0 
        gamma_table = tables["gamma_table"] 
        numerator =  0.0 
        denominator = 0.0
        
        for k in range(len(gamma_table)): 
            for t in range(len(gamma_table[k]) - 1): 
                denominator += gamma_table[k][t][i] 
                if obslist[k][t] == symbol:
                    numerator += gamma_table[k][t][i] 
        bj = numerator / denominator
        if bj == 0:
            bj = threshold
        return bj


    def compute_zi(self, alphas, betas, qi, qj, obs):
        zi = alphas[qi] * self.A[qi][qj] * self.B[qj][obs] * betas[qj]
        return zi
        
    def compute_gamma(self, alphas, betas, qi, ct):
        gam = (alphas[qi] * betas[qi]) / float(ct)
        if gam == 0:
            pass
        return gam

    def create_zi_gamma_tables(self, obslist):      
        zi_table = [] 
        gamma_table = [] 
        for obs in obslist: 
            self.forward(obs)
            self.backward(obs)
            
            zi_t = []
            gamma_t = [] 

            for t in range(len(obs) - 1): 
                zi_t.append({}) 
                gamma_t.append({}) 
                for i in self.states:
                    zi_t[t][i] = {}
                    gamma_t[t][i] = self.compute_gamma(self.fwd_scaled[t], self.bwk_scaled[t], i, self.clist[t])
                    for j in self.states:
                        zi_t[t][i][j] = self.compute_zi(self.fwd_scaled[t], self.bwk_scaled[t + 1], i, j, obs[t + 1])
            zi_table.append(zi_t)
            gamma_table.append(gamma_t)
        return {"zi_table": zi_table, "gamma_table": gamma_table}

    def compute_pi(self, tables, i):
        gamma_table = tables["gamma_table"] 
        numerator = 0.0
        denominator = 0.0

        pi = 0.0
        for k in range(len(gamma_table)): 
            pi += gamma_table[k][0][i] 
        return pi

    def B_W(self, obslist): 
        count = 1
        log_prob = []
        log_prob.append(sum(self.forward(obs) for obs in obslist))
        iteration = 1
        while True:
            sum(self.forward(obs) for obs in obslist)
            tables = self.create_zi_gamma_tables(obslist)            
            temp_A = {}
            temp_B = {}
            temp_pi = {}

            for i in self.states:
                temp_A[i] = {}
                temp_B[i] = {}
                temp_pi[i] = self.compute_pi(tables, i)
                for sym in self.symbols:
                    temp_B[i][sym] = self.compute_bj(tables, i, obslist, sym)
                for j in self.states:
                    temp_A[i][j] = self.compute_aij(tables, i, j)
            normalizer = 0.0
            for v in temp_pi.values():
                normalizer += v
            for k, v in temp_pi.items():
                temp_pi[k] = v / normalizer

            self.A = temp_A
            self.B = temp_B
            self.pi = temp_pi
            
            log_prob.append(sum(self.forward(obs) for obs in obslist)) 
            Delta = (abs(log_prob[iteration]-log_prob[iteration-1]))/abs(log_prob[iteration-1])*100
            iteration = iteration+1
            if Delta < 0.005: break
        
        return (temp_A, temp_B, temp_pi),log_prob


    
    def simulate(self, T):
        def draw_from(probs):
            return np.where(np.random.multinomial(1,probs) == 1)[0][0]
        observations = np.zeros(T, dtype=int)
        states = np.zeros(T, dtype=int)
        
        obs = []
        state = []
        
        states[0] = draw_from(list(self.pi.values()))
        state.append(str(states[0]))
        
        observations[0] = draw_from(list(self.B[state[0]].values()))
        obs.append(str(observations[0]))
        for t in range(1, T):
            list(h.A['0'].values())
            states[t] = draw_from(list(self.A[state[t-1]].values()))
            state.append(str(states[t]))
            observations[t] = draw_from(list(self.B[state[t]].values()))
            obs.append(str(observations[t]))
        return obs,state
    
    def predict(self,obs):
        P = {}
        for k in self.symbols:
            obs_new = obs+[k]
            P[k] = math.exp(self.forward(obs_new))/math.exp(self.forward(obs))
        return P,max(P, key=P.get)

if __name__ == '__main__':
    #load training data
    with open('/Users/heruojin/Desktop/train534.dat') as file:
         observation_list = file.read().split('\n')  
    n = 1000
    for i in range(1000):
        obs = observation_list[i].split(' ')
        observation_list[i] = [obs[i] for i in range(40)]
    observation_list = np.array(observation_list[0:1000])
    kf = KFold(n_splits=10, shuffle=True)
    #CV
    log_p_M = []
    initial =['/Users/heruojin/Desktop/random3.json','/Users/heruojin/Desktop/random4.json',
              '/Users/heruojin/Desktop/random5.json','/Users/heruojin/Desktop/random6.json',
              '/Users/heruojin/Desktop/random7.json','/Users/heruojin/Desktop/random8.json']
    for filename in initial:
        log_p = []
        for train_index, test_index in kf.split(observation_list):
            train, test = observation_list[train_index], observation_list[test_index]
            hmm = MyHmmScaled(filename)
            hmm.forward_backward_multi_scaled(train)
            log_p.append(sum(hmm.forward_scaled(obs) for obs in test))
        log_p_M.append(np.mean(log_p))
    #simulation
    h = Hmm('/Users/heruojin/Desktop/random4.json')
    observations_data, states_data = h.simulate(1000)
    h_train = h.B_W([observations_data])
    obs_train = h.viterbi(observations_data)
    h.forward(observations_data)
    #estimate model
    start = time.time()
    hmm = Hmm('/Users/heruojin/Desktop/random6.json')
    result = hmm.B_W(observation_list[0:500])
    end = time.time()
    #learning curve
    #for trainging 
    log_p=[]
    log_p2 = []
    for i in range(0,500):
        log_p.append(hmm.forward(observation_list[i]))
        log_p2.append(hmm.forward(observation_list[i+500]))

    plt.plot(range(1,67,1), result[1], 'r')
    plt.title('log_likelihood vs. iteration round k ')
    plt.xlabel('ieration')
    plt.ylabel('log_likelihood')
    plt.grid(True)
    plt.show()

    plt.plot(range(1,501,1), log_p, 'k')
    plt.plot(range(1,501,1), log_p2, 'r')
    plt.title('log_likelihood vs. sequence ')
    plt.xlabel('sequence')
    plt.ylabel('log_likehood')
    plt.legend(['c1', 'c2'], loc='upper right')
    plt.grid(True)
    plt.show()

    #predict
    prediction_obs= []
    for i in range(500):
        prediction_obs.append(hmm.predict(list(observation_list[i+500][0:39]))[1])
    true = [observation_list[i+500][-1] for i in range(500)]
    sum([true[i] == prediction_obs[i] for i in range(500)])/500
    
    with open('/Users/heruojin/Desktop/test1_534.dat') as file:
         test = file.read().split('\n')  
    for i in range(50):
        t = test[i].split(' ')
        test[i] = [t[j] for j in range(40)]
    test = np.array(test[0:50])
    
    #forward
    log_p_test = []
    for i in range(50):
        log_p_test.append(hmm.forward(test[i]))
    plt.plot(range(1,51,1), log_p_test, 'k')
    plt.title('log_likelihood vs. sequence ')
    plt.xlabel('sequence')
    plt.ylabel('log_likehood')
    plt.grid(True)
    plt.show()
    
    #sequence of states
    path2 = []
    for t in test:
        path_guess2 = hmm.viterbi(t)
        path2.append(path_guess2[1])
    path_2 = [list(map(int,path2[i])) for i in range(50)]
    
    #prediction
    prediction_test = {}
    for i in range(50):
        prediction_test["seq{}".format(i+1) ] = hmm.predict(list(test[i]))[1]

