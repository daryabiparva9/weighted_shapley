import sys
import time

import pandas as pd
import numpy as np
from itertools import combinations
from shap import Explanation
import sys
import torch
import math
import random
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import math
import xgboost
import shap
import time
import itertools




class Weighted_Shapley:
    def __init__(self, X, y, model, n_features, explanation_type="standard", weights=None, model_type="linear", oracle=None):
        '''
        Oracle for sparsest is a dictionary with keys as features and values as a set of sets. Each set is a set of subsets of features that 
        are not allowed to be in the same ordering as the key feature.
        
        
        
        '''
        self.X = X
        self.y = y
        self.base_value = model.intercept_
        self.feature_names = X.columns.tolist()
        self.model = model
        self.num_outputs = 1
        if n_features > 12:
            raise ValueError("n_features must be less than 13")
        self.n_features = n_features
        self.explanation_type = explanation_type
        if weights is None and explanation_type == "standard":
            self.weights = 1.0 / math.factorial(n_features)
        else:
            self.weights = weights
        self.model_type = model_type
        if oracle is None:
            self.oracle = False
        else:
            self.oracle = oracle
        
    def find_standard_shapley(self, data_point):
        '''
        Consider all orderings
        '''
        perms = list(itertools.permutations(self.feature_names))
        #print(perms)
        phi = pd.DataFrame(columns = self.feature_names)
        order_num = 0
        for ordering in perms:
            z_i = []
            base = self.expected_value(data_point, z_i)
            for variable in ordering:
                new = self.expected_value(data_point, z_i+[variable])
                phi_pi = new - base
                base = new
                phi.loc[order_num, variable] = phi_pi
                z_i.append(variable)
            order_num += 1
        phi_i = phi.mean(axis=0)
        return phi_i
       
    def find_markov_blanket_shapley(self, data_point, parentchild, markov_blanket_oracle):
        if not markov_blanket_oracle:
            raise ValueError("The version with no oracle is not yet implemented")
        s = set(parentchild)
        temp3 = [x for x in self.feature_names if x not in s]
        #temp3 is the set of varibles that are neither parents nor children of the target variable
        perms1 = list(itertools.permutations(parentchild))
        perms2 = list(itertools.permutations(temp3))
        #we need to find all orderings that start with parents and children of the target variable
        #so we get the permutation of all the parents and children of the target variable and all the other variables
        #and also the permutations of the other variables
        #and then we combine them
        mb_perms = []
        from itertools import zip_longest
        for item1 in perms1:
            for item2 in perms2:
                #item1 is the permutation of the parents and children of the target variable
                #item2 is the permutation of the other variables
                temp = item1 + item2
                mb_perms.append(temp)
                
        phi = pd.DataFrame(columns = self.feature_names)
        order_num = 0
        #for comments see sparsest shapley
        for ordering in mb_perms:
            z_i = []
            base = self.expected_value(data_point, z_i)
            for variable in ordering:
                if not markov_blanket_oracle[variable]:
                    new = self.expected_value(data_point, z_i+[variable])
                    phi_pi = new - base
                    base = new
                    phi.loc[order_num, variable] = phi_pi
                    z_i.append(variable)
                elif set(z_i) in markov_blanket_oracle[variable]:
                    phi_pi = 0
                    phi.loc[order_num, variable] = phi_pi
                    z_i.append(variable)
                else:
                    new = self.expected_value(data_point, z_i+[variable])
                    phi_pi = new - base
                    base = new
                    phi.loc[order_num, variable] = phi_pi
                    z_i.append(variable)
            order_num += 1
        zero_num = (phi == 0).astype(int).sum(axis=1)
        max_zeronum = zero_num.max()
        phi = phi[zero_num == max_zeronum]
        phi_i = phi.mean(axis=0)
        return phi_i
    
    def find_sparsest_shapley(self,data_point, sparsest_oracle):
        if not sparsest_oracle:
            raise ValueError("The version with no oracle is not yet implemented")
        perms = list(itertools.permutations(self.feature_names))
        #print(perms)
        phi = pd.DataFrame(columns = self.feature_names)
        order_num = 0
        for ordering in perms:
            z_i = []
            base = self.expected_value(data_point, z_i)
            for variable in ordering:
                #if sparsest_oracle[variable] == {} it means that the variable can not be d-sep from X_{N+1}
                if not sparsest_oracle[variable]:
                    new = self.expected_value(data_point, z_i+[variable])
                    phi_pi = new - base
                    base = new
                    phi.loc[order_num, variable] = phi_pi
                    z_i.append(variable)
                #if z_i is in the set of sparsest_oracle[variable] then the variable is d-sep from X_{N+1} given z_i
                elif set(z_i) in sparsest_oracle[variable]:
                    phi_pi = 0
                    phi.loc[order_num, variable] = phi_pi
                    z_i.append(variable)
                #if not then z_i is not d-seping variable from X_{N+1}
                else:
                    new = self.expected_value(data_point, z_i+[variable])
                    phi_pi = new - base
                    base = new
                    phi.loc[order_num, variable] = phi_pi
                    z_i.append(variable)
            order_num += 1
        #find the number of zeros for each ordering
        zero_num = (phi == 0).astype(int).sum(axis=1)
        #find the maximum number of zeros
        max_zeronum = zero_num.max()
        #only keep orderings with the maximum number of zeros
        phi = phi[zero_num == max_zeronum]
        #find the uniform mean of the orderings with the maximum number of zeros (according to Maximum Entropy principle)
        phi_i = phi.mean(axis=0)
        return phi_i
    
    def find_ancestor_shapley(self, data_point, ancestor_oracle):
        if not ancestor_oracle:
            raise ValueError("The version with no oracle is not yet implemented")
        phi = pd.DataFrame(columns = self.feature_names)
        order_num = 0
        for ordering in ancestor_oracle:
            z_i = []
            base = self.expected_value(data_point, z_i)
            for variable in ordering:
                new = self.expected_value(data_point, z_i+[variable])
                phi_pi = new - base
                base = new
                phi.loc[order_num, variable] = phi_pi
                z_i.append(variable)
            order_num += 1
        phi_i = phi.mean(axis=0)
        return phi_i
      
    def find_shapley(self, data_point, *, sparsest_oracle=False, ancestor_oracle=False, explanation_type="standard"):
        if explanation_type == "standard":
            a = self.find_standard_shapley(data_point)
            return self.r_to_shap_format(a, data_point)
        elif explanation_type == "ancestor":
            return self.find_ancestor_shapley(data_point, ancestor_oracle)
        elif explanation_type == "markov blanket":
            #we first find the set of parents and children of the target variable
            #to do this we need to find the sparsest shapley value
            temp = self.find_sparsest_shapley(data_point, sparsest_oracle)
            nonzero =(temp != 0)
            parentchild = nonzero.index[nonzero].tolist()
            temp_shap = self.find_markov_blanket_shapley(data_point,  parentchild=parentchild, markov_blanket_oracle=sparsest_oracle)
            return self.r_to_shap_format(temp_shap, data_point)
        elif explanation_type == "sparsest":
            temp_shap = self.find_sparsest_shapley(data_point, sparsest_oracle)
            return self.r_to_shap_format(temp_shap, data_point)
        else:
            exit("explanation_type not supported")
    
    def expected_value(self, data_point, subset):
        if self.model_type == "linear":
            if len(subset) == 0:
                return self.model.intercept_
            else:
                lr = LinearRegression()
                lr.fit(self.X[subset], self.y)
                #print(f'for {subset} coef is {lr.coef_}')
                expected_value = lr.predict([data_point[subset]])
                return expected_value
        
    def r_to_shap_format(self, r, data_point):

        base_values = self.base_value
        values = r.to_numpy()
        print(f' data point type is :{type(data_point)}')
        data = data_point.to_numpy()
        print(data.shape[0])
        base_values =np.zeros((data.shape[0],))
        data.reshape([1, data.shape[0]])
        
        
        values = values.reshape([1, values.shape[0]])

        data = np.array([data])

        display_data = None
        instance_names = None
        feature_names = self.feature_names
        output_names = None
        output_indexes = None
        lower_bounds = None
        upper_bounds = None
        main_effects = None
        hierarchical_values = None
        clustering = None

        explanation_list = []
        if self.num_outputs > 0:
            for i in range(self.num_outputs):
                temp_values = values[i]
                temp_base_values = base_values[i]
                temp_data = data[i]
                out = Explanation(temp_values,
                                    temp_base_values,
                                    temp_data,
                                    display_data,
                                    instance_names,
                                    feature_names,
                                    output_names,
                                    output_indexes,
                                    lower_bounds,
                                    upper_bounds,
                                    main_effects,
                                    hierarchical_values,
                                    clustering)
                explanation_list.append(out)

        if len(explanation_list) == 1:
            explanation_list = explanation_list[0]

        return explanation_list

            