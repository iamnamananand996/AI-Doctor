var slider = document.getElementById("myRange");
var output = document.getElementById("blood_pressure");
output.innerHTML = slider.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
slider.oninput = function() {
  output.innerHTML = this.value;
}

var cholesterol_Range = document.getElementById("cholesterol_Range");
var cholesterol_value = document.getElementById("cholesterol_value");

// console.log(cholesterol_Range);

cholesterol_value.innerHTML = cholesterol_Range.value;
cholesterol_Range.oninput = function() {
  cholesterol_value.innerHTML = this.value;
  // console.log('data');
}


var maximum_heart_rate = document.getElementById("maximum_heart_rate");
var maximum_heart_rate_value = document.getElementById("maximum_heart_rate_value");

// console.log(cholesterol_Range);

maximum_heart_rate_value.innerHTML = maximum_heart_rate.value;
maximum_heart_rate.oninput = function() {
  maximum_heart_rate_value.innerHTML = this.value;
  // console.log('data');
}

var oldpeak = document.getElementById("oldpeak");
var oldpeak_value = document.getElementById("oldpeak_value");

// console.log(cholesterol_Range);

oldpeak_value.innerHTML = oldpeak.value;
oldpeak.oninput = function() {
  oldpeak_value.innerHTML = this.value;
  // console.log('data');
}