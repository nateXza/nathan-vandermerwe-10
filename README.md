# nathan-vandermerwe-10
This project demonstrates Linear Regression in Rust using the burn library for computations and textplots for visualization.

Linear Regression in Rust
GitHub License

This repository contains a simple implementation of Linear Regression in Rust. It uses the burn library for numerical computations and the textplots crate for visualization.

Table of Contents
Overview
Features
Prerequisites
Getting Started
Usage
How It Works
Contributing
License
Overview
This project demonstrates how to implement linear regression in Rust step-by-step. It includes generating synthetic data, defining a model, training it, and evaluating its performance. The results are visualized using ASCII plots for easy interpretation.

Features
Synthetic Data Generation : Create a dataset where y=2x+1 with added Gaussian noise.
Linear Regression Model : Define a simple linear regression model using the burn library.
Training Process : Train the model using stochastic gradient descent (SGD) and monitor convergence.
Evaluation : Test the model on unseen data and compute the Mean Squared Error (MSE).
Visualization : Plot the true vs. predicted values using the textplots crate.
Prerequisites
Before running this project, ensure you have the following installed:

Rust : Install Rust via rustup .
Cargo : Comes bundled with Rust. Use it to build and run the project.
Getting Started
Clone the Repository :
bash
Copy
1
2
git clone https://github.com/your-username/linear-regression-rust.git
cd linear-regression-rust
Install Dependencies :
Add the required dependencies to your Cargo.toml:
toml
Copy
1
2
3
4
[dependencies]
burn = "0.4"
rand = "0.8"
textplots = "0.3"
Build and Run :
bash
Copy
1
cargo run
This will generate synthetic data, train the model, and display the results, including an ASCII plot.
Usage
The program generates synthetic data, trains a linear regression model, and evaluates its performance. After running the program, you should see output similar to the following:

Copy
1
2
3
4
5
6
7
8
9
Epoch: 0, Loss: 1.5432
Epoch: 100, Loss: 0.0987
Epoch: 200, Loss: 0.0654
...
Test Loss: 0.0521

       +-------------------------------------------------------------+
     4 |                                            ..................|
       |                                           ...................|
       |                                          ....................|
       |                                         ||||||||||||||||||||||
       |                                        |||||||||||||||||||||||
       |                                       ||||||||||||||||||||||||
       |                                      |||||||||||||||||||||||||
       |                                     ||||||||||||||||||||||||||
       |                                    |||||||||||||||||||||||||||
       |                                   ||||||||||||||||||||||||||||
       |                                  |||||||||||||||||||||||||||||
     0 |..................................|||||||||||||||||||||||||||||
       +-------------------------------------------------------------+
       -3                                                          3
How It Works
Data Generation :
The program generates synthetic data points (x,y) where y=2x+1 with added Gaussian noise.
Model Definition :
A linear regression model is defined with a weight and bias parameter.
The forward pass computes predictions using  
y
^
​
 =XW+b.
The loss function is Mean Squared Error (MSE).
Training :
The model is trained using stochastic gradient descent (SGD).
Gradients are computed via backpropagation, and the optimizer updates the model parameters.
Evaluation :
The trained model is tested on unseen data, and the test loss is computed.
Results are visualized using ASCII plots.
Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
