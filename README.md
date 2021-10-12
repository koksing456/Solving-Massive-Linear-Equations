# Solving-Massive-Linear-Equations
Solve massive linear equation parallelly using OpenMP and CUDA.

# What is a linear equation?
Linear equations are frequently used to compute rates such as speed, time and distance problems. As a standard form of a linear equation in one-variable, this rule has the usual form of ax + b = 0. From this equation, x is the variable, a and b are constants in this equation. A linear equation in two-variable is written in the conventional form as ax + by = c. The variables x and y are variables, whereas the constants are a,b and c.

# Gaussian Elimination
This section will discuss the proposed algorithm, Gaussian Elimination and show how it works by giving an example. An example is shown as below Figure 1.
![image](https://user-images.githubusercontent.com/70791269/136903700-3294a5d6-3618-4d3f-984b-2805a80e30f6.png)
Figure 1. An example of system of linear equations

Based on linear equations given by figure 1, there is a system of linear equations which consists of 3 equations with 3 unknowns that need to be solved. The first step is to transcribe the linear system into an augmented matrix which is shown as below figure 2.
![image](https://user-images.githubusercontent.com/70791269/136903730-bfec1053-b4aa-40ea-a867-a27aa90ce440.png)
Figure 2. Transcribe the system of linear equations into an augmented matrix

The next step is applying Gaussian elimination method to row reducing the augmented matrix, shown as below figure 3.
![image](https://user-images.githubusercontent.com/70791269/136903745-79792110-50bc-4aa8-8f1c-ee8faa3093c2.png)
Figure 3. Row reducing the augmented matrix

The reduced matrix results in its echelon form which is shown as below figure 4.
![image](https://user-images.githubusercontent.com/70791269/136903780-f7a27242-1b45-4185-8518-8c9b9f93b157.png)
Figure 4. Reduced matrix in echelon form

The linear equations then can be observed whether it is solvable for each of the linear equation variables or not solvable at all. At this point the first phase of row reduction which is called Forward elimination is performed completely. The next stage is performing back substitution. From figure 4, we can automatically see that the value of the variable z is z = -2. After finding the value of variable z, then proceed to the second equation to solve variable y, and lastly, substitute y and z to the first equation to solve x. The steps are shown as below figure 5.
![image](https://user-images.githubusercontent.com/70791269/136903821-37aa5fb4-bbd3-45c9-b777-0645b34708b9.png)
Figure 5. Perform back substitution to find value x,y, and z

# Performance Evaluation Techniques 
## Speed Up
Speed up refers to the ratio of the time required to execute a program using a specific algorithm on a device with a single processor (i.e. T (1), n = 1) to the time required to execute the same program using a specific algorithm on a machine with multiple processors which is T(n). Speed up can produce a relative gain in shifting from a sequential algorithm to a parallel algorithm. The formula of the speed up is proposed as S(n) = T(1)/T(n). Figure below shows the relationship between the speed up and number of processors. As we can see from the figure above, the speed up is directly proportional to the number of processors therefore a linear dotted line is depicted. However,  when the overhead occurs, the line becomes sub linear. 
![image](https://user-images.githubusercontent.com/70791269/136904164-1d472947-91ef-49d5-916b-def1e8c33e47.png)
