# Linear-Discriminant-Functions-Perceptron-LMS

Repository contains several experiments conducted in the process of exploring the Linear Discriminant Functions;Perceptron (Margin + Relaxation), LMS

A. PerceptronConvergencetheorem.pdf contains proof for perceptron convergence theorem.

B. q2345.py implements the following:

	B1. Single Sample Perceptron Algorithm.

	B2. Single Sample Perceptron with margin.

	B3. Relaxation algorithm with margin.

	B4. Least Mean Squared (LMS) Rule: Pseudo Inverse

	B5. Widrow-Hoff or Least Mean Squared (LMS) Rule: Gradient Descent

> The data set used for the above two-class problem is

> w1= [(1; 6); (7; 2); (8; 9); (9; 9); (4; 8); (8; 5)]

> w2= [(2; 1); (3; 3); (2; 4); (7; 1); (1; 3); (5; 2)]

C. Adjusting the above dataset so that the solutions of Perceptron and LMS align

D. Modifying the above dataset such that classes are non linearly separable and running B4 and B5 with suitable stopping criteria.

E. Trying out different Initial weight vectors: A good initial heuristic is to start with “Difference of average of positive input vectors and average of negative input vectors”. We can justify that as the final weight vector is a linear combination of input vectors.For given samples the initial weight vector is (0, 2.38, 4.17) and to check convergence dependence on initial weights two other weight vectors can be considered (1, 1, 1) and (300, 200, 100).

>To run all the above experiments there is no need of giving any command line arguments.
