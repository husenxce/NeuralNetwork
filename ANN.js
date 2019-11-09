/*
The MIT License (MIT)

Copyright (c) <2015> <Hussain Mir Ali>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

"use strict";

var Neural_Network = function() {
    this.MathJS = require('mathjs');
    this.q = require('q');
};

Neural_Network.prototype.sigmoid = function(z) {
    var scope = {
            z: (typeof(z) === "number") ? this.MathJS.matrix([
                [z]
            ]) : z
        },
        sigmoid;

    scope.ones = this.MathJS.ones(scope.z.size()[0], scope.z.size()[1]);
    sigmoid = this.MathJS.eval('(ones+(e.^(z.*-1))).^-1', scope); //1/(1+e^(-z))
    return sigmoid;
};

Neural_Network.prototype.forwardPropagation = function(X) {
    var y_result, X = this.MathJS.matrix(X) || this.x, scope = {};
    this.z2 = this.MathJS.multiply(X, this.W1);
    scope.z2 = this.z2;
    this.z2 = this.MathJS.eval('z2', scope);
    this.a2 = this.sigmoid(this.z2);
    this.z3 = this.MathJS.multiply(this.a2, this.W2);
    scope.z3 = this.z3;
    this.z3 = this.MathJS.eval('z3', scope);
    y_result = this.sigmoid(this.z3);
    return y_result;
};

Neural_Network.prototype.sigmoid_Derivative = function(z) {
    var scope = {
            z: z
        },
        sigmoid_Derivative;
    scope.ones = this.MathJS.ones(z.size()[0], z.size()[1]);
    sigmoid_Derivative = this.MathJS.eval('(e.^(z.*-1))./(ones+(e.^(z.*-1))).^2', scope); //(1+e^(-z))/(1+e^(-z))^2

    return sigmoid_Derivative;
};

Neural_Network.prototype.costFunction = function() {
    var J;
    var scope = {};
    this.y_result = this.forwardPropagation(this.x);
    scope.y_result = this.y_result;
    scope.y = this.y;
    scope.x = this.x;

    J = this.MathJS.sum(this.MathJS.eval('0.5*((y-y_result).^2)', scope))/this.x.size()[0];

    return J;
};

Neural_Network.prototype.costFunction_Derivative = function() {
    this.y_result = this.forwardPropagation(this.x);
    var scope = {};
    scope.y_result = this.y_result;
    scope.y = this.y;
    scope.diff = this.MathJS.eval('-(y-y_result)', scope);
    scope.sigmoid_Derivative_z3 = this.sigmoid_Derivative(this.z3);

    var del_3 = this.MathJS.eval('diff.*sigmoid_Derivative_z3', scope);
    var dJdW2 = this.MathJS.multiply(this.MathJS.transpose(this.a2), del_3);

    scope.arrA = this.MathJS.multiply(del_3, this.MathJS.transpose(this.W2));
    scope.arrB = this.sigmoid_Derivative(this.z2);

    var del_2 = this.MathJS.eval('arrA.*arrB',scope);
    var dJdW1 = this.MathJS.multiply(this.MathJS.transpose(this.x), del_2);

    return [dJdW1, dJdW2];

};

Neural_Network.prototype.gradientDescent = function() {
    var gradient = new Array(2),
        cost,
        scope = {},
        defered = this.q.defer(),
        i = 0;
    console.log('Training ...\n');
    while (1) {
        gradient = this.costFunction_Derivative();
        scope.W1 = this.W1;
        scope.W2 = this.W2;
        scope.rate = this.learningRate;
        scope.dJdW1 = gradient[0];
        scope.dJdW2 = gradient[1];
        this.W2 = this.MathJS.eval('W2 - rate*dJdW2', scope);
        this.W1 = this.MathJS.eval('W1 - rate*dJdW1', scope);
        cost = this.costFunction()
        if (cost < (1 / (this.threshold||this.MathJS.exp(6)))) {
            console.log("\nVisit http://www.softnami.com/dailycoding/signup.html to receive a coding question everyday.\n");
            defered.resolve();
            break;
        }
        if (i % 100 === 0) {
            console.log({'iteration': i, 'cost': cost}); //notify cost values for diagnosing the performance of learning algorithm.
        }
        i++;
    }
    return defered.promise;
};

Neural_Network.prototype.train_network = function(learningRate, threshold, X, Y) {
    this.x = this.MathJS.matrix(X);
    this.y = this.MathJS.matrix(Y);
    this.threshold = threshold;

    if ((this.y.size()[0] !== this.x.size()[0])) {
        console.log('\nPlease change the size of the input matrices so that X and Y have same number of rows.');
    }
    else if((this.y.size()[1]>1)){
        console.log('\nPlease change the size of the input matrix Y such that there is only one column because this a single class classifier.');
    } else {
        this.inputLayerSize = this.x.size()[1];
        this.outputLayerSize = 1;
        this.hiddenLayerSize = this.x.size()[1] + 1;
        this.learningRate = learningRate || 0.5;

        this.W1 = (this.MathJS.random(this.MathJS.matrix([this.inputLayerSize, this.hiddenLayerSize]), -5, 5));
        this.W2 = (this.MathJS.random(this.MathJS.matrix([this.hiddenLayerSize, this.outputLayerSize]), -5, 5));
        return this.gradientDescent();
    }
};

Neural_Network.prototype.predict_result = function(X) {
    var y_result = this.forwardPropagation(X);
    return y_result;
};

module.exports = Neural_Network;
