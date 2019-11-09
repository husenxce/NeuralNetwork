# Neural Network
#[Author: Hussain Mir Ali]
An artificial neural network I created with a single hidden layer. This project has been written in JavaScript. The applications include disease prediction, optimizing workout routine and stock prediction.

#Note:
This project uses batch gradient descent so it is best suited for binary classification which has a lower initial cost. But improvements to this algorithm will be made to run stochastic gradient descent.

#Installation:
npm link: https://www.npmjs.com/package/artificial-neural-network

To use the project:

run: npm install -g artificial-neural-network

and follow the sample usage provided below.

#Advanced Neural Network:
An advance version of this Neural Network algorithm is available for $1 USD: http://machine-learning-module.herokuapp.com/  for purchase.

#Sample usage:

```javascript
const Neural_Network = require('artificial-neural-network');
const nn = new Neural_Network();

nn.train_network(0.1, undefined /*optional threshold value*/, [
    [1, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 1]
], [
    [1],
    [0],
    [1],
    [1],
    [0],
    [1],
    [1],
    [0]
]).then(console.log(nn.predict_result([[1,0,0,1,0,1]])));

/*Output
Training ...

{ iteration: 0, cost: 1.383523290363864 }
{ iteration: 100, cost: 0.04008406998951956 }
{ iteration: 200, cost: 0.016181475081737937 }
{ iteration: 300, cost: 0.009841798424077541 }
{ iteration: 400, cost: 0.0069985481625215226 }
{ iteration: 500, cost: 0.005402782030422182 }
{ iteration: 600, cost: 0.00438707375793734 }
{ iteration: 700, cost: 0.003686178233980667 }
{ iteration: 800, cost: 0.003174502735338863 }
{ iteration: 900, cost: 0.0027851304470238596 }
{ iteration: 1000, cost: 0.0024792318930790076 }
{ _data: [ [ 0.030592746473324182 ] ],
  _size: [ 1, 1 ],
  _datatype: undefined }

*/
```
