function handleFormSubmit() {
  var riskTolerance = document.getElementById("risk-tolerance").value;
  var investmentGoals = document.getElementById("investment-goals").value;

  // Make a POST request to the /recommendations endpoint with the risk tolerance and investment goals.
  var xhr = new XMLHttpRequest();
  xhr.open("POST", "/recommendations");
  xhr.setRequestHeader("Content-Type", "application/json");
  xhr.send(JSON.stringify({
    riskTolerance: riskTolerance,
    investmentGoals: investmentGoals
  }));

  // When the response is received, update the feedback text.
  xhr.onload = function() {
    var feedbackText = document.getElementById("feedback");
    feedbackText.innerHTML = "Your recommendations are being generated. Please wait...";
  };
}

// When the form is submitted, call the handleFormSubmit() function.
document.getElementById("form").addEventListener("submit", handleFormSubmit);
