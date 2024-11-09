import * as tf from '@tensorflow/tfjs';

/** Synthetic data for regression **/
const x_train = tf.tensor2d([
  [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]
]);

const y_train = tf.tensor2d([
  [3], [6], [8], [10], [14], [16], [18], [20], [22], [24]
]);

/** Validation data for evaluation **/
const x_val = tf.tensor2d([
  [11], [12], [13]
]);

const y_val = tf.tensor2d([
  [26], [28], [30]
]);

/** Create a more advanced neural network model **/
const model = tf.sequential();

// First hidden layer with ReLU activation function
model.add(tf.layers.dense({
  units: 64,                // 64 neurons
  inputShape: [1],          // Input size is 1 (because we have one feature)
  activation: 'relu'        // Using ReLU activation to introduce non-linearity
}));

// Second hidden layer with ReLU activation function
model.add(tf.layers.dense({
  units: 32,                // 32 neurons
  activation: 'relu'        // ReLU activation
}));

// Output layer, no activation function (for regression)
model.add(tf.layers.dense({
  units: 1                  // Single output value for regression
}));

/** Compile the model **/
model.compile({
  optimizer: 'adam',                // Adam optimizer is more efficient for complex models
  loss: 'meanSquaredError',         // MSE loss function for regression tasks
  metrics: ['mse']                  // Metric: Mean Squared Error (MSE)
});

/** Train the model with early stopping **/
async function train() {
  const earlyStopping = tf.callbacks.earlyStopping({
    monitor: 'val_loss',        // Monitor validation loss
    patience: 5,                // Stop if validation loss doesn't improve after 5 epochs
    verbose: 1
  });

  await model.fit(x_train, y_train, {
    epochs: 100,               // Number of epochs to train
    validationData: [x_val, y_val],   // Validation data to check performance during training
    callbacks: [earlyStopping], // Early stopping callback
    batchSize: 2               // Batch size for training
  });

  console.log('Model trained!');
}

/** Evaluate the model **/
async function evaluate() {
  const result = await model.evaluate(x_val, y_val);
  console.log('Validation Loss:', result[0].dataSync());
}

/** Make a prediction for new data **/
async function predict() {
  const prediction = model.predict(tf.tensor2d([[11]])) as tf.Tensor;
  prediction.print();  // Output: predicted value for x=11
}

/** Run the training, evaluation, and prediction functions **/
train().then(() => {
  evaluate();
  predict();
});
