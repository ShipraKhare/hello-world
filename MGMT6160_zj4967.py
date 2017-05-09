import random;
import matplotlib.pyplot as plt;
import scipy.stats as st;

#Chapter 3: 1: mean 
#Input Parameters - List or Tuple of real numbers
#Output Parameters - mean of real numbers in the list
def mean(input_list):
    return float(sum(input_list)) / max(len(input_list), 1);
    
#Chapter 3: 2: median 
#Input Parameters - List of real numbers
#Output Parameters - median of real numbers in the list
def median(input_list):
    input_list.sort();
    half = len(input_list)//2;
    oddness = len(input_list) % 2;
    if oddness == 1:
        return input_list[half];
    else:
        low = float(input_list[half-1]);
        high = float(input_list[half]);
        return (low + high)/2;
    
#Chapter 3: 3: mode 
#Input Parameters - List of real numbers
#Output Parameters - mode of real numbers in the list
def mode(input_list):
    mode_value = 0;
    mode_dict = dict();
    for input in input_list:
        if input in mode_dict:
            count = mode_dict.get(input);
            count = count + 1;
            mode_dict[input] = count;
        else:
            mode_dict[input] = 1;
    max = 0;
    for key, value in mode_dict.items():
        if value > max:
            max = value;
            mode_value = key ;
    return mode_value;
    
#Chapter 3: 4: Percentile
#Input Parameters - List of real numbers and a real number(p) between 0-100
#Output Parameters - pth percentile of real numbers in the list
def percentile(input_list, p):
    if(p <= 0):
        raise ValueError('pth percentile cannot be less than or equal to 0.');
    elif(p >= 100):
        raise ValueError('pth percentile cannot be greater than or equal to 100.');
    input_list.sort();
    length = len(input_list);
    index = int(round(length * p/100));
    return input_list[index];

#Chapter 3: 5: Quartile
#Input Parameters - List of real numbers and a integer q with values 1,2 or 3
#Output Parameters - qth Quartile of real numbers in the list
def quartile(input_list, q):
    if(q != 1 and q != 2 and q !=3):
        raise ValueError('Invalid Quartile value. Valid values are 1, 2 and 3.');
    if(q == 1):
        return percentile(input_list, 25);
    elif(q == 2):
        return percentile(input_list, 50);
    elif(q == 3):
        return percentile(input_list, 75);

#Chapter 3: 6: num_range
#Input Parameters - List of real numbers
#Output Parameters - range of these numbers
def num_range(input_list):
    input_list.sort();
    length = len(input_list);
    lowest = input_list[0];
    highest = input_list[length - 1];
    return highest - lowest;

#Chapter 3: 7: iqr
#Input Parameters - List of real numbers
#Output Parameters - InterQuartile Range of these numbers
def iqr(input_list):
    q1 = quartile(input_list, 1);
    q3 = quartile(input_list, 3);
    return q3 - q1;

#Chapter 3: 8: var_p
#Input Parameters - List of real numbers
#Output Parameters - Population Variance of these numbers
def var_p(input_list):
    avg = mean(input_list);
    list = [];
    for input in input_list:
        list.append((input - avg) ** 2);
    return mean(list);        
    
#Chapter 3: 9: var_s
#Input Parameters - List of real numbers
#Output Parameters - Sample Variance of these numbers
def var_s(input_list):
    avg = mean(input_list);
    total = 0;
    for input in input_list:
        total += ((input - avg) ** 2);
    return total / (len(input_list) -1) ;
    
#Chapter 3: 10: std_p
#Input Parameters - List of real numbers
#Output Parameters - Population Standard Deviation of these numbers
def std_p(input_list):
    return (var_p(input_list) ** 0.5);
    
#Chapter 3: 11: std_s
#Input Parameters - List of real numbers
#Output Parameters - Sample Standard Deviation of these numbers
def std_s(input_list):
    return (var_s(input_list) ** 0.5);
    

#Chapter 3: 12: cv_p
#Input Parameters - List of real numbers
#Output Parameters - population coefficient of variation of these numbers
def cv_p(input_list):
    sd = std_p(input_list);
    avg = mean(input_list);
    return (sd / avg * 100);
    
#Chapter 3: 13: cv_s
#Input Parameters - List of real numbers
#Output Parameters - sample coefficient of variation of these numbers
def cv_s(input_list):
    sd = std_s(input_list);
    avg = mean(input_list);
    return (sd / avg * 100);
    
#Chapter 3: 14: skewness
#Input Parameters - List of real numbers
#Output Parameters - skewness of these numbers
def skewness(input_list):
    xbar = mean(input_list);
    sd = std_s(input_list);
    n = float(len(input_list));
    lhs = float(n / ((n - 1) * (n - 2)));
    skew = 0;
    for input in input_list:
        skew = skew + (lhs * (((input - xbar) / sd) ** 3));
    return skew;
    
#Chapter 3: 15: z_score
#Input Parameters - List of real numbers and a real number x
#Output Parameters - z-score for x
def z_score(input_list, x):
    return (x - mean(input_list)) / (std_s(input_list));
    
#Chapter 3: 16: outlier_z
#Input Parameters - List of real numbers
#Output Parameters - list of outliers of these numbers
def outlier_z(input_list):
    avg = mean(input_list);
    sd = std_s(input_list);
    outliers = [];
    for input in input_list:
        zs = (input - avg) / sd;
        if(zs > 3 or zs < -3):
            outliers.append(input);

    if(len(outliers) == 0):
        return "No outliers present in the list.";
    else:
        return outliers;
        
    
#Chapter 3: 17: outlier_iqr
#Input Parameters - List of real numbers
#Output Parameters - list of outliers of these numbers
def outlier_iqr(input_list):
    q1 = quartile(input_list, 1);
    q3 = quartile(input_list, 3);
    interQuartile = iqr(input_list);
    low = q1 - (1.5 * interQuartile);
    high = q3 + (1.5 * interQuartile);
    outliers = [];
    for input in input_list:        
        if(input > high or input < low):
            outliers.append(input);

    if(len(outliers) == 0):
        return "No outliers present in the list.";
    else:
        return outliers;
        
#Chapter 3: 18: cov_p
#Input Parameters - Two List of real numbers of equal lengths
#Output Parameters - Population Covariance between these lists
def cov_p(x, y):
    if(len(x) != len(y)):
        raise ValueError('Lists are not equal in length.');
    mean_x = mean(x);
    mean_y = mean(y);    
    total = 0;
    for index in range(len(x)):
        x_input = x[index];
        y_input = y[index];
        total += (x_input - mean_x) * (y_input - mean_y);
    
    return total / len(x);        
    
#Chapter 3: 19: cov_s
#Input Parameters - Two List of real numbers of equal lengths
#Output Parameters - Sample Covariance between these lists
def cov_s(x, y):
    if(len(x) != len(y)):
        raise ValueError('Lists are not equal in length.');
    mean_x = mean(x);
    mean_y = mean(y);    
    total = 0;
    for index in range(len(x)):
        x_input = x[index];
        y_input = y[index];
        total += (x_input - mean_x) * (y_input - mean_y);
    
    return total / (len(x) - 1);        
    
#Chapter 3: 20: r_pearson_p
#Input Parameters - Two List of real numbers of equal lengths
#Output Parameters - Correlation Coefficient between these lists
def r_pearson_p(x, y):
    if(len(x) != len(y)):
        raise ValueError('Lists are not equal in length.');
    covp = cov_p(x, y);
    std_x = std_p(x);
    std_y = std_p(y);
    
    return covp / std_x * std_y;        
    
#Chapter 3: 21: r_pearson_s
#Input Parameters - Two List of real numbers of equal lengths
#Output Parameters - Correlation Coefficient between these lists
def r_pearson_s(x, y):
    if(len(x) != len(y)):
        raise ValueError('Lists are not equal in length.');
    covs = cov_s(x, y);
    std_x = std_s(x);
    std_y = std_s(y);
    
    return covs / std_x * std_y;        
    
#Chapter 4: 1: factorial
#Input Parameters - Non negative integer
#Output Parameters - Factorial of the number
def factorial(num):
    if(num < 0):
        raise ValueError('Number cannot be less than zero.');
    elif(num == 0):
        return 1;
    fact = 1;
    while(num > 0):
        fact = fact * num;
        num = num - 1;
        factorial(num);

    return fact;            
    
#Chapter 4: 2: combination
#Input Parameters - Total number of objects and number of selected objects
#Output Parameters - Number of combination
def combination(N, n):    
    return (factorial(N) / (factorial(n) * factorial(N - n)));
    
    
#Chapter 4: 1: permutation
#Input Parameters - Total number of objects and number of selected objects
#Output Parameters - Number of permutation
def permutation(N, n):    
    return (factorial(N) / (factorial(N - n)));
    
#Chapter 5
def binomial_dist(n, p, x, cdf = True):
    pmf = False;
    if(not cdf):
        pmf = True;
    
    if(pmf):
        #Calculate PMF
        comb = combination(n, x);
        mass = comb * (p ** x) * ((1 - p) ** (n - x));
        return mass;
    elif(cdf):
        #Calculate CDF
        mass = 0;
        while x >= 0:
            comb = combination(n, x);
            mass = mass + (comb * (p ** x) * ((1 - p) ** (n - x)));
            x = x - 1;
        return mass;
        
        
def poisson_dist(mu, x, cdf = True):
    poisson = False;
    if(not cdf):
        poisson = True;
    
    if(poisson):
        #Calculate Poisson
        e = 2.71828; #since e is a constant equal to approximately 2.71828
        return ((e ** (0 - mu)) * (mu ** x) / (factorial(x)));
    elif(cdf):
        #Calculate CDF
        mass = 0;
        e = 2.71828; #since e is a constant equal to approximately 2.71828
        while x >= 0:
            mass = mass + ((e ** (0 - mu)) * (mu ** x) / (factorial(x)));
            x = x - 1;
        return mass;  
              
        
def hypergeometric_dist(N, r, n, x, cdf = True):
    hyper = False;
    if(not cdf):
        hyper = True;
        
    if(hyper):
        #Calculate Hyper Geometric
        return (combination(r, x) * combination((N - r), (n - x)) / combination(N, n));
    elif(cdf):
        #Calculate CDF
        mass = 0;
        while x >= 0:
            mass = mass + (combination(r, x) * combination((N - r), (n - x)) / combination(N, n));
            x = x - 1;
        return mass;
        
        
#Chapter 6        
def uniform_dist(a, b, x, cdf = True):
    uniform = False;
    if(not cdf):
        uniform = True;
        
    if(uniform):
        #Calculate Uniform Dist
        if(a <= x and x <= b):
            return (1 / (b - a));
        else:
            return 0;
    elif(cdf):
        #Calculate CDF
        mass = 0;
        while x >= 0:
            if(a <= x and x <= b):
                mass = mass + (1 / (b - a));
            x = x - 1;
        return mass;


def normal_dist(mu, sigma, x, cdf = True):
    normal = False;
    if(not cdf):
        normal = True;
        
    if(normal):
        #Calculate normal Dist
        e = 2.71828; #since e is a constant equal to approximately 2.71828
        pi = 3.14159;
        exp = (0 - ((x - mu) ** 2) / (2 * (sigma ** 2)));
        return (1 / (sigma * ((2 * pi) ** 0.5)) * (e ** exp));
    elif(cdf):
        #Calculate CDF
        mass = 0;
        e = 2.71828; #since e is a constant equal to approximately 2.71828
        pi = 3.14159;
        while x > 0:
            exp = (0 - ((x - mu) ** 2) / (2 * (sigma ** 2)));
            mass = mass + (1 / (sigma * ((2 * pi) ** 0.5)) * (e ** exp));
            x = x - 1;
        return mass;
        
        
def exponential_dist(mu, x, cdf = True):
    exponential = False;
    if(not cdf):
        exponential = True;
        
    if(exponential):
        #Calculate exponential Dist
        e = 2.71828; #since e is a constant equal to approximately 2.71828
        return (1 / mu * (e ** (0 - (x / mu))));
    elif(cdf):
        #Calculate CDF
        e = 2.71828; #since e is a constant equal to approximately 2.71828
        mass = 0;
        while x > 0:
            mass = mass + (1 / mu * (e ** (0 - (x / mu))));
            x = x - 1;
        return mass;
        

def test_clt_with_uniform_dist(n, t):
    sample_means = [];
    for y in range(0, t):
        samples = [];
        for x in range(0, n):
            samples.append(random.random());
        sample_mean = mean(samples);
        sample_means.append(sample_mean);
    
    print plt.hist(sample_means, bins=100, normed=1);
    grand_mean = mean(sample_means);
    print "Grand Mean = ", grand_mean, "\n";
    std_dev = std_s(sample_means);
    print "Standard Deviation = ", std_dev, "\n";



#Chapter 8

def ci_mean(L, a, sigma = None):
    if sigma is None:
        #Calculate with sigma unknown
        m = mean(L);
        n = len(L);
        s = std_s(L);
        alpha = a;
        tscore = st.t.ppf(1 - alpha / 2, n - 1);
        upper = m + (tscore * s / (n ** 0.5));
        lower = m - (tscore * s / (n ** 0.5));
        return [lower, upper];
    else:
        #Calculate with sigma known
        m = mean(L);
        n = len(L);
        alpha = a;
        zscore = st.norm.ppf(1 - alpha / 2);
        upper = m + (zscore * sigma / (n ** 0.5));
        lower = m - (zscore * sigma / (n ** 0.5));
        return [lower, upper];


def ci_proportion(pbar, n, a):
    alpha = a;
    zscore = abs(st.norm.ppf(alpha / 2));
    rhs = zscore * ((pbar * (1 - pbar) / n) ** 0.5);
    lower = pbar - rhs;
    if(lower < 0):
        lower = 0;
    upper = pbar + rhs;
    return [lower, upper];

#Chapter 9
def hypo_test_for_mean(sample_mean, hypo_mean, n, a, sd, isPopSD, tail = 0):
    alpha = a;
    critical_val = 0;
    test_stats = 0;
    p_value = 0;
    Hzero = "H0 is not rejected";
    if isPopSD is True:
        # z-value
        pop_sd = sd;
        test_stats = (sample_mean - hypo_mean) / (pop_sd / (n ** 0.5));
        if(tail == 0):
            #Two-tailed
            zscore = st.norm.ppf(alpha / 2);
            critical_val = [-zscore, zscore];
            if(test_stats <= 0):
                p_value = (2 * (st.norm.cdf(test_stats)));
            else:
                p_value = (2 * (1 - st.norm.cdf(test_stats)));
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";
        elif(tail == -1):
            #Lower tail
            zscore = st.norm.ppf(alpha);
            critical_val = zscore;
            p_value = st.norm.cdf(test_stats);
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";
        elif(tail == 1):
            #Upper tail
            zscore = st.norm.ppf(1 - alpha);
            critical_val = zscore;
            p_value = (1 - st.norm.cdf(test_stats));
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";
    else:
        # t-value
        sample_sd = sd;
        test_stats = (sample_mean - hypo_mean) / (sample_sd / (n ** 0.5));
        if(tail == 0):
            #Two-tailed
            tscore = st.t.ppf(alpha / 2, n - 1);
            critical_val = [-tscore, tscore];
            if(test_stats <= 0):
                p_value = (2 * (st.t.cdf(test_stats, n - 1)));
            else:
                p_value = (2 * (1 - st.t.cdf(test_stats, n - 1)));
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";
        elif(tail == -1):
            #Lower tail
            tscore = st.t.ppf(alpha, n - 1);
            critical_val = -tscore;
            p_value = st.t.cdf(test_stats, n - 1);
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";
        elif(tail == 1):
            #Upper tail
            tscore = st.t.ppf(1 - alpha, n - 1);
            critical_val = tscore;
            p_value = (1 - st.t.cdf(test_stats, n - 1));
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";
                
    return [test_stats, critical_val, p_value, Hzero];                

def hypo_test_for_proportion(sample_prop, hypo_pop_prop, n, a, tail = 0):
    test_stats = 0;
    critical_val = 0;
    p_value = 0;
    Hzero = "True (H0 is not rejected)";
    alpha = a;
    test_stats = ((sample_prop - hypo_pop_prop) / (((hypo_pop_prop * (1 - hypo_pop_prop)) / n) ** 0.5));
    if(tail == 0):
        #Two-tailed
        zscore = st.norm.ppf(alpha / 2);
        critical_val = [-zscore, zscore];
        if(test_stats <= 0):
            p_value = (2 * st.norm.cdf(test_stats));
        else:
            p_value = (2 * (1 - (st.norm.cdf(test_stats))));
        if(p_value <= alpha):
            Hzero = "False (H0 is rejected)";
        else:
            Hzero = "True (H0 is not rejected)";
    elif(tail == -1):
        #Lower tail
        zscore = st.norm.ppf(alpha);
        critical_val = zscore;
        p_value = st.norm.cdf(test_stats);
        if(p_value <= alpha):
            Hzero = "False (H0 is rejected)";
        else:
            Hzero = "True (H0 is not rejected)";
    elif(tail == 1):
        #Upper tail
        zscore = st.norm.ppf(1 - alpha);
        critical_val = zscore;
        p_value = (1 - st.norm.cdf(test_stats));
        if(p_value <= alpha):
            Hzero = "False (H0 is rejected)";
        else:
            Hzero = "True (H0 is not rejected)";
        
    return [test_stats, critical_val, p_value, Hzero];


def power_in_hypo_test_for_mean(beta_mean, hypo_pop_mean, n, a, sd, isPopSD, tail = 0):

    if isPopSD is True:
        # z-value
        pop_sd = sd;
        test_stats = ((beta_mean - hypo_pop_mean) / (pop_sd / (n ** 0.5)));      
        if(tail == 0):
            #Two-tailed
            if(test_stats <= 0):
                beta = (2 * st.norm.cdf(test_stats));
            else:    
                beta = (2 * (1 - (st.norm.cdf(test_stats))));
            return (1 - beta);
        elif(tail == -1):
            #Lower tail
            beta = st.norm.cdf(test_stats);
            return (1 - beta);
        elif(tail == 1):
            #Upper tail
            beta = (1 - st.norm.cdf(test_stats));
            return (1 - beta);
    else:
        # t-value
        sample_sd = sd;
        test_stats = ((beta_mean - hypo_pop_mean) / (sample_sd / (n ** 0.5)));
        if(tail == 0):
            #Two-tailed
            if(test_stats <= 0):
                beta = (2 * st.t.cdf(test_stats, n - 1));
            else:
                beta = (2 * (1 - st.t.cdf(test_stats, n - 1)));
            return (1 - beta);
        elif(tail == -1):
            #Lower tail
            beta = st.t.cdf(test_stats, n - 1);
            return (1 - beta);
        elif(tail == 1):
            #Upper tail
            beta = (1 - st.t.cdf(test_stats, n - 1));
            return (1 - beta);

#Chapter 10

def ci_for_mean_difference(sample_means, sample_sizes, sds, a, isSampleSD):
    alpha = a;
    z_or_t_score = 0;
    
    x1bar = float(sample_means[0]);
    x2bar = float(sample_means[1]);
    n1 = float(sample_sizes[0]);
    n2 = float(sample_sizes[1]);
    sigma1 = float(sds[0]);
    sigma2 = float(sds[1]);

    if isSampleSD is True:
        nr = (((sigma1 ** 2) / n1 + (sigma2 ** 2) / n2) ** 2);
        dr = ((1 / (n1 - 1)) * (((sigma1 ** 2) / n1) ** 2)) + ((1 / (n2 - 1)) * (((sigma2 ** 2) / n2) ** 2));
        degree_of_freedom = int(nr / dr);
        z_or_t_score = st.t.ppf(1 - alpha / 2, degree_of_freedom);
    else:
        z_or_t_score = st.norm.ppf(1 - alpha / 2);

    r1 = (sigma1 ** 2) / n1;
    r2 = (sigma2 ** 2) / n2;
    r3 = r1 + r2;
    r4 = (r3 ** 0.5);
    rhs = z_or_t_score * r4;
    lhs = x1bar - x2bar;
    
    lower = lhs - rhs;
    upper = lhs + rhs;
    
    return [lower, upper];


def hypo_test_for_mean_difference(sample_means, sample_sizes, sds, a, isSampleSD = True, hypo_diff = 0, tail = 0):

    test_stats = 0;
    critical_val = 0;
    p_value = 0;
    Hzero = "True (H0 is not rejected)";
    
    alpha = a;
    x1bar = float(sample_means[0]);
    x2bar = float(sample_means[1]);
    n1 = float(sample_sizes[0]);
    n2 = float(sample_sizes[1]);

    if isSampleSD is True:
        # t-value
        s1 = float(sds[0]);
        s2 = float(sds[1]);

        nr = float(((s1 ** 2) / n1 + (s2 ** 2) / n2) ** 2);
        dr = ((1 / (n1 - 1)) * (((s1 ** 2) / n1) ** 2)) + ((1 / (n2 - 1)) * (((s2 ** 2) / n2) ** 2));
        degree_of_freedom = int(nr / dr);

        test_stats = ((x1bar - x2bar) - hypo_diff) / ((((s1 ** 2) / n1) + ((s2 ** 2) / n2)) ** 0.5);
        if(tail == 0):
            #Two-tailed
            tscore = st.t.ppf(alpha / 2, degree_of_freedom);
            critical_val = [-tscore, tscore];
            if(test_stats <= 0):
                p_value = 2 * (st.t.cdf(test_stats, degree_of_freedom));
            else:
                p_value = 2 * (1 - (st.t.cdf(test_stats, degree_of_freedom)));
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";
        elif(tail == -1):
            #Lower tail
            critical_val = st.t.ppf(alpha, degree_of_freedom);
            p_value = st.t.cdf(test_stats, degree_of_freedom);
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";
        elif(tail == 1):
            #Upper tail
            critical_val = st.t.ppf(1 - alpha, degree_of_freedom);
            p_value = 1 - st.t.cdf(test_stats, degree_of_freedom);
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";
    else:
        # z-value
        sigma1 = float(sds[0]);
        sigma2 = float(sds[1]);
        
        test_stats = ((x1bar - x2bar) - hypo_diff) / (((sigma1 * sigma1 / n1) + (sigma2 * sigma2 / n2)) ** 0.5);
        if(tail == 0):
            #Two-tailed
            zscore = st.norm.ppf(alpha / 2);
            critical_val = [-zscore, zscore];
            if(test_stats <= 0):
                p_value = 2 * st.norm.cdf(test_stats);
            else:
                p_value = 2 * (1 - st.norm.cdf(test_stats));
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";
        elif(tail == -1):
            #Lower tail
            critical_val = st.norm.ppf(alpha);
            p_value = st.norm.cdf(test_stats);
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";
        elif(tail == 1):
            #Upper tail
            critical_val = st.norm.ppf(1 - alpha);
            p_value = 1 - st.norm.cdf(test_stats);
            if(p_value <= alpha):
                Hzero = "False (H0 is rejected)";
            else:
                Hzero = "True (H0 is not rejected)";

    return [test_stats, critical_val, p_value, Hzero];         


def ci_for_proportion_difference(sample_props, sample_sizes, a):
    pbar1 = sample_props[0];
    pbar2 = sample_props[1];
    n1 = sample_sizes[0];
    n2 = sample_sizes[1];
    alpha = a;
    zscore = abs(st.norm.ppf(alpha / 2));
    lhs = pbar1 - pbar2;
    rhs = zscore * (((pbar1 * (1 - pbar1) / n1) + (pbar2 * (1 - pbar2) / n2)) ** 0.5);
    lower = lhs - rhs;
    upper = lhs + rhs;
    
    return [lower, upper];


def hypo_test_for_proportion_difference(sample_props, sample_sizes, a, tail = 0):
    test_stats = 0;
    critical_val = 0;
    p_value = 0;
    Hzero = "True (H0 is not rejected)";
    
    pbar1 = float(sample_props[0]);
    pbar2 = float(sample_props[1]);
    n1 = float(sample_sizes[0]);
    n2 = float(sample_sizes[1]);
    alpha = a;
    
    pbar = ((n1 * pbar1) + (n2 * pbar2)) / (n1 + n2);
    test_stats = (pbar1 - pbar2) / ((pbar * (1 - pbar) * ((1 / n1) + (1 / n2))) ** 0.5);
    if(tail == 0):
        #Two-tailed
        zscore = st.norm.ppf(alpha / 2);
        critical_val = [-zscore, zscore];
        if(test_stats <= 0):
            p_value = 2 * (st.norm.cdf(test_stats));
        else:
            p_value = 2 * (1 - st.norm.cdf(test_stats));
        if(p_value <= alpha):
            Hzero = "False (H0 is rejected)";
        else:
            Hzero = "True (H0 is not rejected)";
    elif(tail == -1):
        #Lower tail
        critical_val = st.norm.ppf(alpha);
        p_value = st.norm.cdf(test_stats);
        if(p_value <= alpha):
            Hzero = "False (H0 is rejected)";
        else:
            Hzero = "True (H0 is not rejected)";
    elif(tail == 1):
        #Upper tail
        critical_val = st.norm.ppf(1 - alpha);
        p_value = 1 - st.norm.cdf(test_stats);
        if(p_value <= alpha):
            Hzero = "False (H0 is rejected)";
        else:
            Hzero = "True (H0 is not rejected)";
    
    return [test_stats, critical_val, p_value, Hzero];
   

#Chapter 11

def ci_for_population_var(sample_var, n, a):
    alpha = a;
    chi2 = st.chi2.ppf((alpha / 2), (n - 1));
    chi1 = st.chi2.ppf((1 - (alpha / 2)), (n - 1));
    lower = (n - 1) * sample_var / chi1;
    upper = (n - 1) * sample_var / chi2;

    return [lower, upper];


def hypo_test_for_population_var(sample_var, hypo_var, n, a, tail):
    
    test_stats = 0;
    critical_val = 0;
    p_value = 0;
    Hzero = "True (H0 is not rejected)";
    
    ssquare = sample_var;
    sigmasquare = hypo_var;
    
    alpha = a;
    
    test_stats = (n - 1) * ssquare / sigmasquare;
    if(tail == -1):
        #Lower tail
        critical_val = st.chi2.ppf(alpha, n - 1);
        p_value = st.chi2.cdf(test_stats, n - 1);
        if(p_value <= alpha):
            Hzero = "False (H0 is rejected)";
        else:
            Hzero = "True (H0 is not rejected)";
    elif(tail == 1):
        #Upper tail
        critical_val = st.chi2.ppf(1 - alpha, n - 1);
        p_value = 1 - st.chi2.cdf(test_stats, n - 1);
        if(p_value <= alpha):
            Hzero = "False (H0 is rejected)";
        else:
            Hzero = "True (H0 is not rejected)";
    
    return [test_stats, critical_val, p_value, Hzero];


def hypo_test_for_two_population_var(sample_vars, sample_sizes, a, tail):
    
    test_stats = 0;
    critical_val = 0;
    p_value = 0;
    Hzero = "True (H0 is not rejected)";
    
    var1 = float(sample_vars[0]);
    var2 = float(sample_vars[1]);
    n1 = float(sample_sizes[0]);
    n2 = float(sample_sizes[1]);
    alpha = a;
    
    test_stats = var1 / var2;
    
    if(tail == -1):
        #Lower tail
        critical_val = st.f.ppf(alpha, n1 - 1, n2 - 1);
        p_value = st.f.cdf(test_stats, n1 - 1, n2 - 1);
        if(p_value <= alpha):
            Hzero = "False (H0 is rejected)";
        else:
            Hzero = "True (H0 is not rejected)";
    elif(tail == 1):
        #Upper tail
        critical_val = st.f.ppf(1 - alpha, n1 - 1, n2 - 1);
        p_value = 1 - st.f.cdf(test_stats, n1 - 1, n2 - 1);
        if(p_value <= alpha):
            Hzero = "False (H0 is rejected)";
        else:
            Hzero = "True (H0 is not rejected)";
    
    return [test_stats, critical_val, p_value, Hzero];
