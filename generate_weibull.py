from scipy.stats import weibull_min

c_shape = 3  # Shape 
loc_location = 0  # Location 
scale_lambda = 50  # Scale  

weibull_dist = weibull_min(c=c_shape, loc=loc_location, scale=scale_lambda)
samples = weibull_dist.rvs(size=20000)

with open("./Data/weibull.txt", "w") as file:
    for s in samples:
        if s <= 100:
            file.write(str(s/100)+"\n")
