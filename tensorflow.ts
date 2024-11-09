import * as tf from '@tensorflow/tfjs';

/** Tensor: A multi-dimensional array. In this case, tensor1d means 1-dimensional. **/

/** This is the input data (independent variable) **/
const x_train = tf.tensor1d([1, 2, 3, 4, 5]);
/** This is the target/output data (dependent variable) **/
const y_train = tf.tensor1d([3, 5, 7, 9, 11]);



/** Create a simple model **/
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

/** Compile the model **/
model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError',
});

/** Train the model **/
async function train() {
  await model.fit(x_train, y_train, { epochs: 100 });
  console.log('Model trained!');
}

/** Make a prediction **/
async function predict() {
  const prediction = model.predict(tf.tensor1d([6])) as tf.Tensor;
  prediction.print();  // Output: predicted value for x=6
}

train().then(() => {
  predict();
});
