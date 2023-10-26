const fs = require("fs");
const path = require("path");

const filePath = path.join(__dirname, "../data/xyz_selected_data.json");
const file = fs.readFileSync(filePath, "utf-8");
const data = JSON.parse(file);
const labels = new Set();

const extractedData = data.reduce((acc, item) => {
  //delete '인프라분류', '세부분류키워드' from item
  if (item["스마트 인프라"] !== null && item["인프라 분류"] !== null) {
    delete item["인프라분류"];
    delete item["세부분류키워드"];
    return acc.concat(item);
  } else {
    return acc;
  }
}, []);

console.log(`extractedData.length: ${extractedData.length}`);
console.log(`extractedData samples: ${JSON.stringify(extractedData[0])}`);
const extractedDataString = JSON.stringify(extractedData);
console.log("extraction completed!");
console.log(labels);

// save the extracted data to a new file
fs.writeFileSync(
  path.join(__dirname, "../data/xyz_selected_data_extracted.json"),
  extractedDataString
);
console.log('saved to "xyz_selected_data_extracted.json"');
