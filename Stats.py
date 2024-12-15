#Q1>Q= Generate a list of 100 integers containing values between 90 to 130 and store it in the variable `int_list`.
#After generating the list, find the following:

  

 # (i) Write a Python function to calculate the mean of a given list of numbers.

#Create a function to find the median of a list of numbers.

  

 # (ii) Develop a program to compute the mode of a list of integers.

  

 # (iii) Implement a function to calculate the weighted mean of a list of values and their corresponding weights.

  

  #(iv) Write a Python function to find the geometric mean of a list of positive numbers.

  

 # (v) Create a program to calculate the harmonic mean of a list of values.

  

#  (vi) Build a function to determine the midrange of a list of numbers (average of the minimum and maximum).

  

 # (vii) Implement a Python program to find the trimmed mean of a list, excluding a certain percentage of
#outliers.

import random

# Generate a list of 100 integers between 90 and 130
int_list = [random.randint(90, 130) for _ in range(100)]
print(f"Generated List: {int_list}")

#(i)To calculae mean and median
from collections import Counter
from functools import reduce
import math
from scipy.stats import trim_mean

def calculate_mean(numbers):
    return sum(numbers) / len(numbers)

mean = calculate_mean(int_list)
print(f"Mean: {mean}")

# Function to calculate the median of a list of numbers
def calculate_median(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    mid = n // 2

    if n % 2 == 0:
        return (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
    else:
        return sorted_numbers[mid]

median = calculate_median(int_list)
print(f"Median: {median}")

# (ii) Program to compute the mode of a list of integers
def calculate_mode(numbers):
    count = Counter(numbers)
    max_count = max(count.values())
    mode = [num for num, freq in count.items() if freq == max_count]
    return mode

mode = calculate_mode(int_list)
print(f"Mode: {mode}")

# (iii) Function to calculate the weighted mean of a list
def calculate_weighted_mean(values, weights):
    weighted_sum = sum(value * weight for value, weight in zip(values, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight

# Example weights (100 random weights between 1 and 10)
weights = [random.randint(1, 10) for _ in range(100)]
weighted_mean = calculate_weighted_mean(int_list, weights)
print(f"Weighted Mean: {weighted_mean}")

# (iv) Function to find the geometric mean of a list of positive numbers
def calculate_geometric_mean(numbers):
    product = reduce(lambda x, y: x * y, numbers)
    return product ** (1 / len(numbers))

geometric_mean = calculate_geometric_mean([num for num in int_list if num > 0])
print(f"Geometric Mean: {geometric_mean}")

# (v) Function to calculate the harmonic mean of a list of numbers
def calculate_harmonic_mean(numbers):
    return len(numbers) / sum(1 / num for num in numbers)

harmonic_mean = calculate_harmonic_mean([num for num in int_list if num > 0])
print(f"Harmonic Mean: {harmonic_mean}")

# (vi) Function to determine the midrange of a list of numbers
def calculate_midrange(numbers):
    return (min(numbers) + max(numbers)) / 2

midrange = calculate_midrange(int_list)
print(f"Midrange: {midrange}")

# (vii) Function to find the trimmed mean of a list
def calculate_trimmed_mean(numbers, proportion):
    return trim_mean(numbers, proportion)

trimmed_mean = calculate_trimmed_mean(int_list, 0.05)  # Trims 5% from both ends
print(f"Trimmed Mean (5% trimmed): {trimmed_mean}")

#Q2> 2. Generate a list of 500 integers containing values between 200 to 300 and store it in the variable `int_list2`.
#After generating the list, find the following:


 # (i) Compare the given list of visualization for the given data:

    

  #  1. Frequency & Gaussian distribution

   # 2. Frequency smoothened KDE plot

    #3. Gaussian distribution & smoothened KDE plot


 # (ii) Write a Python function to calculate the range of a given list of numbers.


  #(iii) Create a program to find the variance and standard deviation of a list of numbers.


  #iv) Implement a function to compute the interquartile range (IQR) of a list of values.


#  (v) Build a program to calculate the coefficient of variation for a dataset.

  

 # (vi) Write a Python function to find the mean absolute deviation (MAD) of a list of numbers.


  #(vii) Create a program to calculate the quartile deviation of a list of values.

  

 # (viii) Implement a function to find the range-based coefficient of dispersion for a dataset.
 
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Step 1: Generate a list of 500 integers between 200 and 300
int_list2 = [random.randint(200, 300) for _ in range(500)]
print(f"Generated List (first 20 values): {int_list2[:20]}\n")

# (i) Visualization
def visualize_data(data):
    # Plot Frequency Distribution and Gaussian Distribution
    plt.figure(figsize=(12, 8))

    # Plot the histogram (frequency distribution)
    sns.histplot(data, bins=20, kde=False, color='skyblue', label='Frequency Distribution', stat='density')

    # Plot the Gaussian Distribution
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(min(data), max(data), 1000)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r-', label=f'Gaussian Distribution (μ={mu:.2f}, σ={sigma:.2f})')

    plt.title('Frequency Distribution with Gaussian Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Plot Frequency Distribution with Smoothened KDE Plot
    plt.figure(figsize=(12, 8))
    sns.histplot(data, bins=20, kde=True, color='lightgreen', label='Frequency & KDE')
    plt.title('Frequency Distribution with Smoothened KDE Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Plot Gaussian Distribution with Smoothened KDE Plot
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data, color='blue', label='Smoothened KDE')
    plt.plot(x, norm.pdf(x, mu, sigma), 'r-', label=f'Gaussian Distribution (μ={mu:.2f}, σ={sigma:.2f})')
    plt.title('Gaussian Distribution with Smoothened KDE Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Call the visualization function
visualize_data(int_list2)

# (ii) Function to calculate the range of a given list of numbers
def calculate_range(numbers):
    return max(numbers) - min(numbers)

range_value = calculate_range(int_list2)
print(f"Range: {range_value}")

# (iii) Program to find the variance and standard deviation
def calculate_variance(numbers):
    return np.var(numbers)

def calculate_std_dev(numbers):
    return np.std(numbers)

variance = calculate_variance(int_list2)
std_dev = calculate_std_dev(int_list2)
print(f"Variance: {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")

# (iv) Function to compute the interquartile range (IQR)
def calculate_iqr(numbers):
    q3, q1 = np.percentile(numbers, [75, 25])
    return q3 - q1

iqr = calculate_iqr(int_list2)
print(f"Interquartile Range (IQR): {iqr}")

# (v) Program to calculate the coefficient of variation
def calculate_coefficient_of_variation(numbers):
    mean = np.mean(numbers)
    std_dev = np.std(numbers)
    return (std_dev / mean) * 100

cv = calculate_coefficient_of_variation(int_list2)
print(f"Coefficient of Variation: {cv:.2f}%")

# (vi) Function to find the mean absolute deviation (MAD)
def calculate_mad(numbers):
    mean = np.mean(numbers)
    return np.mean([abs(x - mean) for x in numbers])

mad = calculate_mad(int_list2)
print(f"Mean Absolute Deviation (MAD): {mad:.2f}")

# (vii) Program to calculate the quartile deviation
def calculate_quartile_deviation(numbers):
    q3, q1 = np.percentile(numbers, [75, 25])
    return (q3 - q1) / 2

quartile_deviation = calculate_quartile_deviation(int_list2)
print(f"Quartile Deviation: {quartile_deviation:.2f}")

# (viii) Function to find the range-based coefficient of dispersion
def calculate_range_coefficient_of_dispersion(numbers):
    return (max(numbers) - min(numbers)) / (max(numbers) + min(numbers))

range_dispersion = calculate_range_coefficient_of_dispersion(int_list2)
print(f"Range-based Coefficient of Dispersion: {range_dispersion:.4f}")


#Q3> Write a Python class representing a discrete random variable with methods to calculate its expected
#value and variance.
class DiscreteRandomVariable:
    def __init__(self, outcomes, probabilities):
      
        if len(outcomes) != len(probabilities):
            raise ValueError("Outcomes and probabilities must have the same length.")
        if abs(sum(probabilities) - 1) > 1e-6:
            raise ValueError("Probabilities must sum to 1.")

        self.outcomes = outcomes
        self.probabilities = probabilities

    def expected_value(self):
        
        return sum(o * p for o, p in zip(self.outcomes, self.probabilities))

    def variance(self):
       
        mean = self.expected_value()
        return sum(p * (o - mean) ** 2 for o, p in zip(self.outcomes, self.probabilities))


# Example Usage
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6  # Uniform distribution for a fair six-sided die

die = DiscreteRandomVariable(outcomes, probabilities)
print(f"Expected Value: {die.expected_value():.2f}")
print(f"Variance: {die.variance():.2f}")


#Q4>Implement a program to simulate the rolling of a fair six-sided die and calculate the expected value and
#variance of the outcomes.
import random

def simulate_die_rolls(n):
  
    outcomes = [random.randint(1, 6) for _ in range(n)]
    mean = sum(outcomes) / n
    variance = sum((x - mean) ** 2 for x in outcomes) / n
    return outcomes, mean, variance

# Simulate 1000 rolls
n_rolls = 1000
rolls, mean, variance = simulate_die_rolls(n_rolls)

print(f"Simulated {n_rolls} rolls of a fair six-sided die.")
print(f"Mean (Expected Value): {mean:.2f}")
print(f"Variance: {variance:.2f}")


#Q5>Create a Python function to generate random samples from a given probability distribution (e.g.,
#binomial, Poisson) and calculate their mean and variance.
import numpy as np

def generate_binomial_samples(n, p, size):
   
    samples = np.random.binomial(n, p, size)
    mean = np.mean(samples)
    variance = np.var(samples)
    return samples, mean, variance

def generate_poisson_samples(lam, size):
   
    samples = np.random.poisson(lam, size)
    mean = np.mean(samples)
    variance = np.var(samples)
    return samples, mean, variance

# Binomial distribution: n = 10 trials, p = 0.5, 1000 samples
binomial_samples, binomial_mean, binomial_variance = generate_binomial_samples(10, 0.5, 1000)
print(f"Binomial Distribution - Mean: {binomial_mean:.2f}, Variance: {binomial_variance:.2f}")

# Poisson distribution: lambda = 5, 1000 samples
poisson_samples, poisson_mean, poisson_variance = generate_poisson_samples(5, 1000)
print(f"Poisson Distribution - Mean: {poisson_mean:.2f}, Variance: {poisson_variance:.2f}")



#Q6> Write a Python script to generate random numbers from a Gaussian (normal) distribution and compute
#the mean, variance, and standard deviation of the samples.
def generate_normal_samples(mean, std_dev, size):
    
    samples = np.random.normal(mean, std_dev, size)
    sample_mean = np.mean(samples)
    sample_variance = np.var(samples)
    sample_std_dev = np.std(samples)
    return samples, sample_mean, sample_variance, sample_std_dev

# Generate 1000 samples from a normal distribution with mean = 0, std_dev = 1
normal_samples, normal_mean, normal_variance, normal_std_dev = generate_normal_samples(0, 1, 1000)

print(f"Normal Distribution - Mean: {normal_mean:.2f}, Variance: {normal_variance:.2f}, Standard Deviation: {normal_std_dev:.2f}")

# Optional: Plot the samples
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.histplot(normal_samples, bins=30, kde=True, color='skyblue')
plt.title("Gaussian (Normal) Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

#Q8>Use seaborn library to load tips dataset. Find the following from the dataset for the columns total_bill
#and tip`:

  

 # (i) Write a Python function that calculates their skewness.


 # (ii) Create a program that determines whether the columns exhibit positive skewness, negative skewness, or is
#approximately symmetric.


 # (iii) Write a function that calculates the covariance between two columns.


  #(iv) Implement a Python program that calculates the Pearson correlation coefficient between two columns.


  #(v) Write a script to visualize the correlation between two specific columns in a Pandas DataFrame using
#scatter plots.

import seaborn as sns
import pandas as pd

# Load the tips dataset
tips = sns.load_dataset('tips')

# Display the first 5 rows
print(tips.head())

#2
from scipy.stats import skew

def calculate_skewness(data):
   
    return skew(data)

# Calculate skewness for 'total_bill' and 'tip' columns
total_bill_skewness = calculate_skewness(tips['total_bill'])
tip_skewness = calculate_skewness(tips['tip'])

print(f"Skewness of total_bill: {total_bill_skewness:.2f}")
print(f"Skewness of tip: {tip_skewness:.2f}")

def skewness_type(skew_value):
    """
    Determine the type of skewness based on the skew value.
    """
    if skew_value > 0:
        return "Positive Skewness"
    elif skew_value < 0:
        return "Negative Skewness"
    else:
        return "Approximately Symmetric"

# Determine skewness type for 'total_bill' and 'tip'
total_bill_skew_type = skewness_type(total_bill_skewness)
tip_skew_type = skewness_type(tip_skewness)

print(f"'total_bill' column exhibits: {total_bill_skew_type}")
print(f"'tip' column exhibits: {tip_skew_type}")


def calculate_covariance(column1, column2):
    """
    Calculate the covariance between two columns.
    """
    return column1.cov(column2)

# Calculate covariance between 'total_bill' and 'tip'
covariance = calculate_covariance(tips['total_bill'], tips['tip'])
print(f"Covariance between total_bill and tip: {covariance:.2f}")

def calculate_pearson_correlation(column1, column2):
    """
    Calculate the Pearson correlation coefficient between two columns.
    """
    return column1.corr(column2)

# Calculate Pearson correlation between 'total_bill' and 'tip'
pearson_correlation = calculate_pearson_correlation(tips['total_bill'], tips['tip'])
print(f"Pearson correlation coefficient between total_bill and tip: {pearson_correlation:.2f}")


import matplotlib.pyplot as plt

# Scatter plot to visualize correlation between 'total_bill' and 'tip'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex')
plt.title('Scatter Plot of Total Bill vs Tip')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.grid(True)
plt.show()

#Q8>Write a Python function to calculate the probability density function (PDF) of a continuous random
#variable for a given normal distribution

import math

def normal_pdf(x, mean, std_dev):
   
    coefficient = 1 / (std_dev * math.sqrt(2 * math.pi))
    exponent = math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
    return coefficient * exponent

# Example usage
x_value = 1.0
mean = 0.0
std_dev = 1.0

pdf_value = normal_pdf(x_value, mean, std_dev)
print(f"PDF value for x = {x_value}, mean = {mean}, std_dev = {std_dev}: {pdf_value:.5f}")

#Q9>.Create a program to calculate the cumulative distribution function (CDF) of exponential distribution

from scipy.stats import norm

# Calculate PDF using scipy
pdf_value_scipy = norm.pdf(x_value, mean, std_dev)
print(f"PDF value using scipy for x = {x_value}, mean = {mean}, std_dev = {std_dev}: {pdf_value_scipy:.5f}")

#Q10>Write a Python function to calculate the probability mass function (PMF) of Poisson distribution
import math

def poisson_pmf(k, lam):
  
    if k < 0:
        raise ValueError("k must be a non-negative integer.")
    if lam <= 0:
        raise ValueError("Lambda (rate parameter) must be a positive value.")
    
    return (lam ** k * math.exp(-lam)) / math.factorial(k)

# Example usage
k_value = 3
lambda_value = 2.5

pmf_value = poisson_pmf(k_value, lambda_value)
print(f"PMF value for k = {k_value}, lambda = {lambda_value}: {pmf_value:.5f}")

#Q11>A company wants to test if a new website layout leads to a higher conversion rate (percentage of visitors
#who make a purchase). They collect data from the old and new layouts to compare.


#To generate the data use the following command:

#```python

#import numpy as np

# 50 purchases out of 1000 visitors

#old_layout = np.array([1] * 50 + [0] * 950)

# 70 purchases out of 1000 visitors  

#new_layout = np.array([1] * 70 + [0] * 930)

#  ```

#Apply z-test to find which layout is successful


from statsmodels.stats.proportion import proportions_ztest

# Data generation
old_layout = np.array([1] * 50 + [0] * 950)  # 50 purchases out of 1000 visitors
new_layout = np.array([1] * 70 + [0] * 930)  # 70 purchases out of 1000 visitors

# Number of successes (purchases) and total visitors for each layout
successes = np.array([old_layout.sum(), new_layout.sum()])
samples = np.array([len(old_layout), len(new_layout)])

# Perform two-proportion z-test
z_stat, p_value = proportions_ztest(successes, samples, alternative='larger')

# Output the results
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The new layout has a significantly higher conversion rate.")
else:
    print("Fail to reject the null hypothesis: No significant difference in conversion rates.")


#Q12>A tutoring service claims that its program improves students' exam scores. A sample of students who
#participated in the program was taken, and their scores before and after the program were recorded
#Use the below code to generate samples of respective arrays of marks:

#```python

#before_program = np.array([75, 80, 85, 70, 90, 78, 92, 88, 82, 87])

#after_program = np.array([80, 85, 90, 80, 92, 80, 95, 90, 85, 88])

#```

#Use z-test to find if the claims made by tutor are true or false.


from scipy.stats import norm

# Data generation
before_program = np.array([75, 80, 85, 70, 90, 78, 92, 88, 82, 87])
after_program = np.array([80, 85, 90, 80, 92, 80, 95, 90, 85, 88])

# Calculate the differences
differences = after_program - before_program

# Calculate the mean and standard deviation of the differences
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)  # Sample standard deviation (ddof=1)
n = len(differences)

# Calculate the z-score
z_score = mean_diff / (std_diff / np.sqrt(n))

# Calculate the p-value (one-tailed test)
p_value = 1 - norm.cdf(z_score)

# Output the results
print(f"Mean of Differences: {mean_diff:.4f}")
print(f"Standard Deviation of Differences: {std_diff:.4f}")
print(f"Z-Score: {z_score:.4f}")
print(f"P-Value: {p_value:.4f}")

# Conclusion based on significance level alpha = 0.05
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The tutoring program significantly improves exam scores.")
else:
    print("Fail to reject the null hypothesis: No significant improvement in exam scores.")

#Q13>A pharmaceutical company wants to determine if a new drug is effective in reducing blood pressure. They
#conduct a study and record blood pressure measurements before and after administering the drug.
#Use the below code to generate samples of respective arrays of blood pressure:


#```python

#before_drug = np.array([145, 150, 140, 135, 155, 160, 152, 148, 130, 138])

#after_drug = np.array([130, 140, 132, 128, 145, 148, 138, 136, 125, 130])

 # ```


#Implement z-test to find if the drug really works or not.


# Data generation
before_drug = np.array([145, 150, 140, 135, 155, 160, 152, 148, 130, 138])
after_drug = np.array([130, 140, 132, 128, 145, 148, 138, 136, 125, 130])

# Calculate the differences
differences = after_drug - before_drug

# Calculate the mean and standard deviation of the differences
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)  # Sample standard deviation (ddof=1)
n = len(differences)

# Calculate the z-score
z_score = mean_diff / (std_diff / np.sqrt(n))

# Calculate the p-value (one-tailed test for reduction in blood pressure)
p_value = norm.cdf(z_score)

# Output the results
print(f"Mean of Differences: {mean_diff:.4f}")
print(f"Standard Deviation of Differences: {std_diff:.4f}")
print(f"Z-Score: {z_score:.4f}")
print(f"P-Value: {p_value:.4f}")

# Conclusion based on significance level alpha = 0.05
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The drug significantly reduces blood pressure.")
else:
    print("Fail to reject the null hypothesis: No significant reduction in blood pressure.")


#Q14>A customer service department claims that their average response time is less than 5 minutes. A sample
#of recent customer interactions was taken, and the response times were recorded.


#Implement the below code to generate the array of response time:

#```python

#response_times = np.array([4.3, 3.8, 5.1, 4.9, 4.7, 4.2, 5.2, 4.5, 4.6, 4.4])

#```

#Implement z-test to find the claims made by customer service department are tru or false



# Data generation
response_times = np.array([4.3, 3.8, 5.1, 4.9, 4.7, 4.2, 5.2, 4.5, 4.6, 4.4])

# Hypothesized mean (5 minutes)
mu_0 = 5

# Sample mean and standard deviation
sample_mean = np.mean(response_times)
sample_std = np.std(response_times, ddof=1)  # Sample standard deviation (ddof=1)
n = len(response_times)

# Calculate the z-score
z_score = (sample_mean - mu_0) / (sample_std / np.sqrt(n))

# Calculate the p-value (one-tailed test)
p_value = norm.cdf(z_score)

# Output the results
print(f"Sample Mean: {sample_mean:.4f}")
print(f"Sample Standard Deviation: {sample_std:.4f}")
print(f"Z-Score: {z_score:.4f}")
print(f"P-Value: {p_value:.4f}")

# Conclusion based on significance level alpha = 0.05
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The average response time is significantly less than 5 minutes.")
else:
    print("Fail to reject the null hypothesis: No significant evidence that the average response time is less than 5 minutes.")



#Q15>U A company is testing two different website layouts to see which one leads to higher click-through rates.
#Write a Python function to perform an A/B test analysis, including calculating the t-statistic, degrees of
#freedom, and p-value.


#Use the following data:

#```python

layout_a_clicks = [28, 32, 33, 29, 31, 34, 30, 35, 36, 37]

layout_b_clicks = [40, 41, 38, 42, 39, 44, 43, 41, 45, 47]

import numpy as np
from scipy.stats import t

def ab_test_analysis(layout_a_clicks, layout_b_clicks):
    # Convert lists to numpy arrays
    layout_a = np.array(layout_a_clicks)
    layout_b = np.array(layout_b_clicks)
    
    # Calculate sample means
    mean_a = np.mean(layout_a)
    mean_b = np.mean(layout_b)
    
    # Calculate sample standard deviations
    std_a = np.std(layout_a, ddof=1)  # Sample standard deviation
    std_b = np.std(layout_b, ddof=1)
    
    # Sample sizes
    n_a = len(layout_a)
    n_b = len(layout_b)
    
    # Calculate t-statistic
    t_statistic = (mean_a - mean_b) / np.sqrt((std_a ** 2) / n_a + (std_b ** 2) / n_b)
    
    # Calculate degrees of freedom
    numerator = ((std_a ** 2) / n_a + (std_b ** 2) / n_b) ** 2
    denominator = ((std_a ** 2) / n_a) ** 2 / (n_a - 1) + ((std_b ** 2) / n_b) ** 2 / (n_b - 1)
    degrees_of_freedom = numerator / denominator
    
    # Calculate the p-value (two-tailed test)
    p_value = 2 * t.sf(np.abs(t_statistic), degrees_of_freedom)
    
    # Output the results
    print(f"Mean of Layout A Clicks: {mean_a:.2f}")
    print(f"Mean of Layout B Clicks: {mean_b:.2f}")
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"Degrees of Freedom: {degrees_of_freedom:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    # Conclusion based on significance level alpha = 0.05
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference between Layout A and Layout B.")
    else:
        print("Fail to reject the null hypothesis: No significant difference between Layout A and Layout B.")

# Data


# Run the A/B test analysis
ab_test_analysis(layout_a_clicks, layout_b_clicks)



#Q16>A pharmaceutical company wants to determine if a new drug is more effective than an existing drug in
#reducing cholesterol levels. Create a program to analyze the clinical trial data and calculate the tstatistic and p-value for the treatment effect.


#use the following data of cholestrol level:

#```python

existing_drug_levels = [180, 182, 175, 185, 178, 176, 172, 184, 179, 183]

new_drug_levels = [170, 172, 165, 168, 175, 173, 170, 178, 172, 176]

from scipy.stats import t

def analyze_drug_effect(existing_drug_levels, new_drug_levels):
    # Convert lists to numpy arrays
    existing = np.array(existing_drug_levels)
    new = np.array(new_drug_levels)
    
    # Calculate sample means
    mean_existing = np.mean(existing)
    mean_new = np.mean(new)
    
    # Calculate sample standard deviations
    std_existing = np.std(existing, ddof=1)  # Sample standard deviation
    std_new = np.std(new, ddof=1)
    
    # Sample sizes
    n_existing = len(existing)
    n_new = len(new)
    
    # Calculate t-statistic
    t_statistic = (mean_existing - mean_new) / np.sqrt((std_existing ** 2) / n_existing + (std_new ** 2) / n_new)
    
    # Calculate degrees of freedom
    numerator = ((std_existing ** 2) / n_existing + (std_new ** 2) / n_new) ** 2
    denominator = ((std_existing ** 2) / n_existing) ** 2 / (n_existing - 1) + ((std_new ** 2) / n_new) ** 2 / (n_new - 1)
    degrees_of_freedom = numerator / denominator
    
    # Calculate the p-value (one-tailed test)
    p_value = t.sf(t_statistic, degrees_of_freedom)
    
    # Output the results
    print(f"Mean of Existing Drug Levels: {mean_existing:.2f}")
    print(f"Mean of New Drug Levels: {mean_new:.2f}")
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"Degrees of Freedom: {degrees_of_freedom:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    # Conclusion based on significance level alpha = 0.05
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: The new drug is significantly more effective than the existing drug.")
    else:
        print("Fail to reject the null hypothesis: No significant difference between the new and existing drugs.")



# Run the analysis
analyze_drug_effect(existing_drug_levels, new_drug_levels)

 #Q17>A school district introduces an educational intervention program to improve math scores. Write a Python
#function to analyze pre- and post-intervention test scores, calculating the t-statistic and p-value to
#determine if the intervention had a significant impact.


#Use the following data of test score:


#  ```python

pre_intervention_scores = [80, 85, 90, 75, 88, 82, 92, 78, 85, 87]

post_intervention_scores = [90, 92, 88, 92, 95, 91, 96, 93, 89, 93]
  

from scipy import stats

def analyze_intervention_effect(pre_scores, post_scores):
    # Convert lists to numpy arrays
    pre = np.array(pre_scores)
    post = np.array(post_scores)
    
    # Calculate the differences between pre and post scores
    differences = post - pre
    
    # Calculate the t-statistic and p-value for the paired t-test
    t_statistic, p_value = stats.ttest_rel(pre, post)
    
    # Output the results
    print(f"Mean of Pre-Intervention Scores: {np.mean(pre):.2f}")
    print(f"Mean of Post-Intervention Scores: {np.mean(post):.2f}")
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    # Conclusion based on significance level alpha = 0.05
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: The intervention significantly improved the test scores.")
    else:
        print("Fail to reject the null hypothesis: The intervention did not significantly improve the test scores.")

# Run the analysis
analyze_intervention_effect(pre_intervention_scores, post_intervention_scores)

#Q18>An HR department wants to investigate if there's a gender-based salary gap within the company. Develop
#a program to analyze salary data, calculate the t-statistic, and determine if there's a statistically
#significant difference between the average salaries of male and female employees.


#Use the below code to generate synthetic data:



# Generate synthetic salary data for male and female employees

np.random.seed(0)  # For reproducibility

male_salaries = np.random.normal(loc=50000, scale=10000, size=20)

female_salaries = np.random.normal(loc=55000, scale=9000, size=20)

import numpy as np
from scipy import stats

# Generate synthetic salary data for male and female employees
np.random.seed(0)  # For reproducibility

male_salaries = np.random.normal(loc=50000, scale=10000, size=20)
female_salaries = np.random.normal(loc=55000, scale=9000, size=20)

def analyze_salary_gap(male_salaries, female_salaries):
    # Calculate the t-statistic and p-value for the two-sample t-test
    t_statistic, p_value = stats.ttest_ind(male_salaries, female_salaries, equal_var=False)
    
    # Output the results
    print(f"Mean of Male Salaries: {np.mean(male_salaries):.2f}")
    print(f"Mean of Female Salaries: {np.mean(female_salaries):.2f}")
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    # Conclusion based on significance level alpha = 0.05
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in salaries between male and female employees.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in salaries between male and female employees.")

# Run the analysis
analyze_salary_gap(male_salaries, female_salaries)

#Q19>A manufacturer produces two different versions of a product and wants to compare their quality scores.
#Create a Python function to analyze quality assessment data, calculate the t-statistic, and decide
#whether there's a significant difference in quality between the two versions.


#Use the following data:



version1_scores = [85, 88, 82, 89, 87, 84, 90, 88, 85, 86, 91, 83, 87, 84, 89, 86, 84, 88, 85, 86, 89, 90, 87, 88, 85]

version2_scores = [80, 78, 83, 81, 79, 82, 76, 80, 78, 81, 77, 82, 80, 79, 82, 79, 80, 81, 79, 82, 79, 78, 80, 81, 82]


from scipy import stats

# Sample data for quality scores of two versions of the product
version1_scores = [85, 88, 82, 89, 87, 84, 90, 88, 85, 86, 91, 83, 87, 84, 89, 86, 84, 88, 85, 86, 89, 90, 87, 88, 85]
version2_scores = [80, 78, 83, 81, 79, 82, 76, 80, 78, 81, 77, 82, 80, 79, 82, 79, 80, 81, 79, 82, 79, 78, 80, 81, 82]

def analyze_quality_scores(version1_scores, version2_scores):
    # Calculate the t-statistic and p-value for the two-sample t-test
    t_statistic, p_value = stats.ttest_ind(version1_scores, version2_scores, equal_var=False)
    
    # Output the results
    print(f"Mean of Version 1 Scores: {np.mean(version1_scores):.2f}")
    print(f"Mean of Version 2 Scores: {np.mean(version2_scores):.2f}")
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    # Conclusion based on significance level alpha = 0.05
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in quality between the two versions.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in quality between the two versions.")

# Run the analysis
analyze_quality_scores(version1_scores, version2_scores)

#Q20>A restaurant chain collects customer satisfaction scores for two different branches. Write a program to
#analyze the scores, calculate the t-statistic, and determine if there's a statistically significant difference in
#customer satisfaction between the branches.


#Use the below data of scores:



branch_a_scores = [4, 5, 3, 4, 5, 4, 5, 3, 4, 4, 5, 4, 4, 3, 4, 5, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 3, 4, 5, 4]

branch_b_scores = [3, 4, 2, 3, 4, 3, 4, 2, 3, 3, 4, 3, 3, 2, 3, 4, 4, 3, 2, 3, 4, 3, 2, 4, 3, 3, 4, 2, 3, 4, 3]


# Sample data for customer satisfaction scores of two branches
branch_a_scores = [4, 5, 3, 4, 5, 4, 5, 3, 4, 4, 5, 4, 4, 3, 4, 5, 5, 4, 3, 4, 5, 4, 3, 5, 4, 4, 5, 3, 4, 5, 4]
branch_b_scores = [3, 4, 2, 3, 4, 3, 4, 2, 3, 3, 4, 3, 3, 2, 3, 4, 4, 3, 2, 3, 4, 3, 2, 4, 3, 3, 4, 2, 3, 4, 3]

def analyze_customer_satisfaction(branch_a_scores, branch_b_scores):
    # Perform two-sample t-test (assuming unequal variances)
    t_statistic, p_value = stats.ttest_ind(branch_a_scores, branch_b_scores, equal_var=False)
    
    # Output the results
    print(f"Mean of Branch A Scores: {np.mean(branch_a_scores):.2f}")
    print(f"Mean of Branch B Scores: {np.mean(branch_b_scores):.2f}")
    print(f"T-Statistic: {t_statistic:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    # Conclusion based on significance level alpha = 0.05
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference in customer satisfaction between the two branches.")
    else:
        print("Fail to reject the null hypothesis: There is no significant difference in customer satisfaction between the two branches.")

# Run the analysis
analyze_customer_satisfaction(branch_a_scores, branch_b_scores)


#Q21>A political analyst wants to determine if there is a significant association between age groups and voter
#preferences (Candidate A or Candidate B). They collect data from a sample of 500 voters and classify
#them into different age groups and candidate preferences. Perform a Chi-Square test to determine if
#there is a significant association between age groups and voter preferences.


#Use the below code to generate data:

np.random.seed(0)

age_groups = np.random.choice([ 18 30 , 31 50 , 51+', 51+'], size=30)

voter_preferences = np.random.choice(['Candidate A', 'Candidate B'], size=30)

from scipy.stats import chi2_contingency

# Generate dataSS
np.random.seed(0)

# Age groups and voter preferences
age_groups = np.random.choice(['18-30', '31-50', '51+'], size=500)
voter_preferences = np.random.choice(['Candidate A', 'Candidate B'], size=500)

# Create a DataFrame
data = pd.DataFrame({'Age Group': age_groups, 'Voter Preference': voter_preferences})

# Create a contingency table
contingency_table = pd.crosstab(data['Age Group'], data['Voter Preference'])

# Perform Chi-Square test
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

# Output the results
print("Contingency Table:")
print(contingency_table)
print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
print(f"P-Value: {p_value:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies:\n{expected}")

# Conclusion based on significance level alpha = 0.05
alpha = 0.05
if p_value < alpha:
    print("\nReject the null hypothesis: There is a significant association between age groups and voter preferences.")
else:
    print("\nFail to reject the null hypothesis: There is no significant association between age groups and voter preferences.")

#Q23>A company implemented an employee training program to improve job performance (Effective, Neutral,
#Ineffective). After the training, they collected data from a sample of employees and classified them based
#on their job performance before and after the training. Perform a Chi-Square test to determine if there is a
#significant difference between job performance levels before and after the training.


#Sample data:



# Sample data: Job performance levels before (rows) and after (columns) training

data = np.array([[50, 30, 20], [30, 40, 30], [20, 30, 40]])

from scipy.stats import chi2_contingency

# Sample data: Job performance levels before (rows) and after (columns) training
data = np.array([[50, 30, 20], [30, 40, 30], [20, 30, 40]])

# Perform Chi-Square test
chi2_stat, p_value, dof, expected = chi2_contingency(data)

# Output the results
print("Contingency Table:")
print(data)
print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
print(f"P-Value: {p_value:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies:\n{expected}")

# Conclusion based on significance level alpha = 0.05
alpha = 0.05
if p_value < alpha:
    print("\nReject the null hypothesis: There is a significant difference in job performance levels before and after the training.")
else:
    print("\nFail to reject the null hypothesis: There is no significant difference in job performance levels before and after the training.")
