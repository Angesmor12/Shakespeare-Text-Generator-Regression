let normalizationData = null;
let allow = 1
let SentimentValue = document.querySelector(".generated-text")
let loadingImage = document.querySelector(".loading-image-container")

let session = ""
let loadPath = ""

function divideIntoChunks(flatTensor, seqLength, vocabSize) {

  const chunks = [];
  for (let i = 0; i < seqLength; i++) {
      const start = i * vocabSize;
      const end = start + vocabSize;
      chunks.push(flatTensor.slice(start, end));
  }

  return chunks;
}

function time(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function predict(inputFeatures, path, key, vocabSize) {

  session = await ort.InferenceSession.create(path);
  
  const max_length = 8;

  let processedFeatures;
  if (inputFeatures.length > max_length) {
      processedFeatures = inputFeatures.slice(0, max_length);
  } else {
      processedFeatures = inputFeatures.concat(new Array(max_length - inputFeatures.length).fill(0));
  }

  const input = new BigInt64Array(processedFeatures.map(item => BigInt(item)));

  const tensor = new ort.Tensor('int64', input, [1, max_length]);

  const feeds = {};
  feeds[key] = tensor;

  const result = await session.run(feeds);

  const logits = divideIntoChunks(result.output.data, 8, vocabSize);

  const probabilities = logits.map((logit, index) => {
    const maxIndex = getMaxIndex(logit); 
    return maxIndex;
  });

  loadPath = path

  return probabilities;
}

function getMaxIndex(probabilities) {
  return probabilities.indexOf(Math.max(...probabilities));
}

async function loadNormalizationInfo(file) {
  if (normalizationData) {
    return normalizationData;
  }

  const response = await fetch('./models/' + file);
  normalizationData = await response.json(); 
  return normalizationData;
}

async function normalizeInputs(text) {

  const dictionary = await loadNormalizationInfo(
    document.querySelector(".algorithm-input").selectedOptions[0].getAttribute("data-json")
  );

  const textArray = text.split(" ");

  const textArrayNormalize = [];

  textArray.forEach((item) => {

    if (dictionary[item]) {
      textArrayNormalize.push(dictionary[item]);
    } else {
      // <UNK> token
      textArrayNormalize.push(1);
    }
  });

  return textArrayNormalize;
}

function formatValue(value) {

  if (value.includes('.')) {
    return false;
  }

  const valueStr = String(value).replace(/,/g, ''); 

  if (valueStr.length > 3) {
    return `${valueStr.slice(0, -3)}.${valueStr.slice(-3)}`;
  } else {
    return `0.${valueStr.padStart(3, '0')}`;
  }
}

function generateRandomArray(size, maxValue) {
    const array = [];
    for (let i = 0; i < size; i++) {
      array.push(Math.floor(Math.random() * maxValue));
    }
    return array;
}

const selectElement = document.querySelector('.algorithm-input');
let size = "14423"
let dataJson = "vocab_dataset_l.json"

selectElement.addEventListener('change', function() {

  const selectedOption = selectElement.options[selectElement.selectedIndex];
  size = selectedOption.getAttribute('data-size');
  dataJson = selectedOption.getAttribute('data-json');
});

async function translateOutput(data, vocab){

  let vocabJson = await loadNormalizationInfo(vocab)

  let text = ""

  data.map((item)=>{
    text = vocabJson[item] + " " + text
  })

  return text
}

document.querySelector('.button-generate').addEventListener('click', async (e) => {
  e.preventDefault()

  if (allow == 1){
    
  allow = 0  
  SentimentValue.classList.add("hidden")
  loadingImage.classList.remove("hidden")
  SentimentValue.classList.remove("typewriter")

  let noise = generateRandomArray(8,size)

  const algorithm = document.querySelector(".algorithm-input").value

  let normalizePrediction = await predict(noise, algorithm, "input", parseInt(size))

  let wordsOutput = await translateOutput(normalizePrediction, dataJson)

  document.querySelector(".generated-text").innerHTML = wordsOutput

  loadingImage.classList.add("hidden")
  SentimentValue.classList.remove("hidden")
  SentimentValue.classList.add("typewriter")

  allow = 1
  
}
});



