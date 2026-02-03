# **Binary Classification with Cross-Entropy**
This project serves as an introduction to Machine Learning by training a basic neural network to perform binary classification. The goal is to predict whether a given input belongs to Class 0 or Class 1.

## **Network Architecture**
The model consists of a simple neural network with:
- **Input Layer:** 1 input node.
- **Hidden Layer:** Multiple neurons using the ReLU activation function.
- **Output Layer:** 1 output node using the Sigmoid activation function for binary probability.

## **how it works**
**1. Data Processing :** We utilize a training set **$(x_{train}, y_{train})$**, where **$x$** represents the input and **$y$** represents the ground truth label (0 or 1).
**2. The Hidden Layer (Forward Pass) :**
   Each input is passed to the hidden layer where:
   - We compute the linear transformation: **$z = (x \cdot \omega_0) + \text{bias}_0$**.
   - We apply the ReLU (Rectified Linear Unit) activation function: **$f(z) = \max(0, z)$**
   - Purpose: ReLU introduces non-linearity allowing the network to decide whether a neuron should be (activate) or remain inactive.
**3. Output Generation :** The outputs from the hidden layer are multiplied by weight **$\omega_1$** and added to **$\beta_1$** (the bias) The final result **$f(x)$** is passed through the Sigmoid Function:
 **$$\sigma(f(x)) = \frac{1}{1 + e^{-f(x)}}$$** , This maps any real-valued number into a probability between 0 and 1
<img src="project/visualisation/model.png" width="400" alt="Neural Network Architecture"> 

## **Optimization & Likelihood**
The core of this project is finding the optimal $\beta_1$. We use the following statistical approach:
**- Bernoulli Distribution:** Since the output is binary, we model the probability using $P(y|\hat{y}) = \hat{y}^y \cdot (1-\hat{y})^{(1-y)}$.
**- Likelihood Function:** We calculate the product of these probabilities for all data points.
<img src="project/visualisation/likelhood.png" width="400" alt="Likelihood Plot">
**- Log-Likelihood:** Because multiplying many probabilities results in numbers too small for computers to handle (underflow) we apply a Logarithm to turn products into sums.
**- Negative Log-Likelihood (NLL):** We multiply by $-1$ because most optimization algorithms are designed to minimize error rather than maximize success
<img src="project/visualisation/Nll.png" width="400" alt="NLL Plot">

## **Conclusion & Limitations**
In this experiment, we found an "optimal" **$\beta_1 \approx 1.5$** by fixing other parameters and searching for the best bias. However, visualizing the results showed a suboptimal global fit
<img src="project/visualisation/beta1_testing_optimality.png" width="400" alt="beta1_test">
**Key Takeaway:** The found $\beta_1$ represents a **local optimum** Because we only tuned one parameter while keeping weights fixed the model lacks the flexibility to fit all data points perfectly.
**Next Steps:** To reach the Global Minimum of the cost function a more robust approach is required: training all weights and biases simultaneously using the **Gradient Descent** algorithm.

   
   
