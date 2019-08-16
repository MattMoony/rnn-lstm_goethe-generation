# RNN (LSTM) - Goethe Generation
_Simple RNN generating Goethe Poems_

---

## About

This is yet another little test project. I thought of doing this, in order to gain more experience in working with Recurrent Neural Networks (especially with "Long Short Term Memory"-Gates). 

## To-Do

* [x] Dataset preparation
* [x] Simple weight initialization
* [x] Advanced weight initialization (Xavier initialization, etc.)
* [x] Simple bias initialization
* [x] Fully connected layer/s
* [x] Activation functions (sigmoid, tanh, softmax)
* [ ] Loss-Function (Cross-Entropy)
* [ ] Gradient computation function
* [ ] Stochastic Mini Batch Gradient Descent
* [ ] Advanced SGD (Momentum, Adam, etc.)
* [ ] J/Epoch-Graph
* [x] Prediction function
* _... probably more to come ..._

## Results

Sadly, I appear to be having very acute high-bias problems. At the moment I'm unsure as to whether this is due to my choice of hyper-parameters or because of a misfunction in my gradient computation. In order to fix this, first, I'll re-check the code and if I can't find any mistakes, I'll write some functions to help me pick the hyper-parameters which allow for the best convergence.

![J/Epoch-Graph(bad convergence)](media/JEpoch_Figure0.png)
_J/Epoch-Graph displaying the bad convergence_

---

... MattMoony (August, 2019)