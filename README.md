# DeepPokerNet

In this project, we designed and compared two models, Master AI and DeepPokerNet that play No limit Texas Hold’em to fully understand the differences between models in an incomplete information game.

## Master AI
Master AI is rule-based. It calculates the winning rate based on the cards in its hands and the cards on the table, and adjust the winning rate by checking the history of the opponent's bids, as the basis
for the action decision. To improve the performance, we use the Bitmap data structure, 200 times faster than other C language programs using the array data structure, and 5000 times faster than the
program written in Python.

## DeepPokerNet
In DeepPokerNet, We trainined deep neural networks using Policy Gradient with RNN and Reward Normalization. We combined three submodels to achieve better performance as follow.
1. Prediction network: predict opponent’s action
2. Action network: generate my action
3. Output network: output action probability
![DeepPokerNet Architecture](assets/DeepPokerNet.png?raw=true "Title")
