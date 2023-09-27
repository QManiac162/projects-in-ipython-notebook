# Cosine similarity example 
'''
Cosine similarity is a mathematical approach used to measure the similarity between pairs of vectors or rows of numbers treated as vectors. It involves representing each value in a sample endpoint coordinates for a vector, with the other endpoint at the origin of the coordinate system. This process is repeated for two samples, and the cosine between the vectors is computed in an m-dimensional space, where m represents the number of values in each sample. The cosine similarity is such that identical vectors have a similarity of 1 since the cosine of 0 is 1. As the vectors become more dissimilar, the cosine value approaches 0, indicating a lower similarity.
'''
'''
The dot product of vectors A (x1​, x2​, x3​) and B (y1​, y2​, y3​), denoted as A.B, can be computed using the following formula:

A.B = x1​∗y1​ + x2​∗y2 ​+ x3​∗y3 ​

and ||A|| & ||B|| is calculated by:

∣∣A∣∣ = ✓(​x1²​+x2​²+x3​²)

∣∣B∣∣ = ✓(​y1²​+y2​²+y3​²)
'''
'''
if we can comprehend a concept in two dimensions, we can extend that understanding to any number of dimensions
'''
import numpy as np
import pandas as pd
# defining 3 vectors
A = [1,2]
B = [2,3]
C = [3,1]

# calculate dot products
ab = np.dot(A,B)
bc = np.dot(B,C)
ca = np.dot(C,A)

# calculate the length of the vector
a = np.linalg.norm(A)
b = np.linalg.norm(B)
c = np.linalg.norm(C)

# calculate the cosen similarity for each pair
sim_ab = ab/(a*b)
sim_bc = bc/(b*c)
sim_ca = ca/(c*a)

# check similarities
print(sim_ab) 
print(sim_bc) 
print(sim_ca)

# utilizing scikit=learn for not doing manual cosine-similarity
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity([A, B, C]))

'''
 The cosine_similarity function returns a matrix of cosine similarities, similar to a correlation matrix. Each row and column in the matrix represents a vector, resulting in a diagonal matrix where the diagonal values are always one (since the vectors are being compared to themselves).
From the obtained results, it becomes apparent that vector A is more similar to vector B compared to vector C. Similarly, vector B is closer to vector C.
'''

print(pd.DataFrame(cosine_similarity([A, B, C]), columns=['A', 'B', 'C'], index=['A', 'B', 'C']))