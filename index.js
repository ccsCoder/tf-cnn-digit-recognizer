import { MnistData } from "./data.js";
import quotes from "./quotes.js";

function setBackgroundWithText(time) {
  const filtered = Object.keys(quotes).filter((key) =>
    key.startsWith(`0${time}`)
  );
  const randomIndex = filtered[Math.floor(Math.random() * filtered.length)];
  const finalQuote = quotes[randomIndex][0];
  console.log(finalQuote);

  const { time: textTime, prefix, book, author, suffix } = finalQuote;

  const quoteHTML = `
    <p class="quote">${prefix} <span class="emphasis">${textTime}</span> ${suffix}</p>
    <p class="author">- ${author} in ${book}</p>
  `;

  document.querySelector(".recognized-digit").innerHTML = quoteHTML;
}

async function saveModel(model) {
  await model.save("localstorage://my-model");
}

async function loadModel() {
  try {
    const model = await tf.loadLayersModel("localstorage://my-model");
    return model;
  } catch (error) {
    console.log("No saved model found, training a new one.");
    return null;
  }
}

function getModel() {
  const model = tf.sequential();

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  // In the first layer of our convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    })
  );

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add(
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: "varianceScaling",
      activation: "softmax",
    })
  );

  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function train(model, data) {
  const metrics = ["loss", "val_loss", "acc", "val_acc"];
  const container = {
    name: "Model Training",
    tab: "Model",
    styles: { height: "1000px" },
  };

  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 55000;
  const TEST_DATA_SIZE = 10000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks,
  });
}

async function showExamples(data) {
  // Create a container in the visor
  const surface = tfvis
    .visor()
    .surface({ name: "Input Data Examples", tab: "Input Data" });

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = "margin: 4px;";
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

const enableRecognition = () => {
  document.querySelector("#clearBtn").disabled = false;
  document.querySelector("#recognizeBtn").disabled = false;
  document.querySelector("h1").textContent = "Draw a digit!";
};

async function run() {
  const data = new MnistData();
  await data.load();

  let model = await loadModel();
  if (!model) {
    model = getModel();
    await showExamples(data);
    tfvis.show.modelSummary(
      { name: "Model Architecture", tab: "Model" },
      model
    );
    await train(model, data);
    await saveModel(model);
    await showAccuracy(model, data);
    await showConfusion(model, data);
  }

  enableRecognition();
}

document.addEventListener("DOMContentLoaded", run);

document.addEventListener("DOMContentLoaded", function () {
  const canvas = document.getElementById("drawingCanvas");
  canvas.willReadFrequently = true;
  const ctx = canvas.getContext("2d");
  const clearBtn = document.getElementById("clearBtn");
  const recognizeBtn = document.getElementById("recognizeBtn");

  // Drawing state
  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;

  // Set up canvas
  ctx.fillStyle = "white";
  ctx.strokeStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 10;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getCoordinates(e);
  }

  function draw(e) {
    if (!isDrawing) return;

    const [currentX, currentY] = getCoordinates(e);

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();

    [lastX, lastY] = [currentX, currentY];
  }

  function stopDrawing() {
    isDrawing = false;
  }

  function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches && e.touches[0]) {
      return [
        (e.touches[0].clientX - rect.left) * scaleX,
        (e.touches[0].clientY - rect.top) * scaleY,
      ];
    }

    return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY];
  }

  function handleTouchStart(e) {
    e.preventDefault();
    startDrawing(e);
  }

  function handleTouchMove(e) {
    e.preventDefault();
    draw(e);
  }

  function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 12;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // clear the .recognized-digit element
    document.querySelector(".recognized-digit").innerHTML = "";
  }

  async function preprocessCanvas() {
    // Create a temporary canvas for resizing
    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");
    tempCanvas.width = 28;
    tempCanvas.height = 28;

    // Clear the tempCanvas before drawing
    tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Scale down the image
    tempCtx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 28, 28);

    // Get the image data
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;

    // Convert to grayscale and normalize
    const input = new Float32Array(28 * 28);
    for (let i = 0; i < data.length; i += 4) {
      // Proper grayscale conversion using luminance weights
      const grayscale =
        0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      // Normalize and invert (MNIST uses white digits on black background)
      input[i / 4] = (255 - grayscale) / 255.0;
    }

    // For debugging - you can visualize the processed image
    // Comment this section out when not needed
    for (let i = 0; i < data.length; i += 4) {
      const value = input[i / 4] * 255;
      data[i] = value; // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255; // A
    }
    tempCtx.putImageData(imageData, 0, 0);

    // Remove the old tempCanvas if it exists
    const oldTempCanvas = document.getElementById("tempCanvas");
    if (oldTempCanvas) {
      document.body.removeChild(oldTempCanvas);
    }

    // Append the new tempCanvas and set its id
    tempCanvas.id = "tempCanvas";
    // document.body.appendChild(tempCanvas); // This will show the processed image

    // Reshape to match MNIST input shape [1, 28, 28, 1]
    return tf.tensor(input).reshape([1, 28, 28, 1]);
  }

  async function recognizeDrawing() {
    try {
      const model = await loadModel();
      const tensor = await preprocessCanvas();

      const predictions = await model.predict(tensor).data();
      console.log(predictions);
      console.log("Max = ", Math.max(...predictions));

      // Get the index of the highest confidence
      const predictedClass = predictions.indexOf(Math.max(...predictions));

      console.log("prediction = ", predictedClass, classNames[predictedClass]);
      await setBackgroundWithText(predictedClass);

      // Free memory
      tensor.dispose();
    } catch (error) {
      console.error("Error during recognition:", error);
    }
  }

  // Drawing event listeners
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mouseout", stopDrawing);

  // Touch support
  canvas.addEventListener("touchstart", handleTouchStart);
  canvas.addEventListener("touchmove", handleTouchMove);
  canvas.addEventListener("touchend", stopDrawing);

  // Button event listeners
  clearBtn.addEventListener("click", clearCanvas);
  recognizeBtn.addEventListener("click", recognizeDrawing);
});

const classNames = [
  "Zero",
  "One",
  "Two",
  "Three",
  "Four",
  "Five",
  "Six",
  "Seven",
  "Eight",
  "Nine",
];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([
    testDataSize,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    1,
  ]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}

async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = { name: "Accuracy", tab: "Evaluation" };
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = { name: "Confusion Matrix", tab: "Evaluation" };
  tfvis.render.confusionMatrix(container, {
    values: confusionMatrix,
    tickLabels: classNames,
  });

  labels.dispose();
}
