function startPrediction() {
    alert("Brain Tumor Prediction will start soon!");
}
// Optional: Add click event listener for debugging or analytics
document.querySelectorAll('.portfolio-link').forEach(link => {
    link.addEventListener('click', (event) => {
        console.log(`Redirecting to: ${event.target.href}`);
        // Optional: Add analytics tracking or prevent default behavior
    });
});

// Ensure Owl Carousel works (if not already included)
$(document).ready(function () {
    $(".gallery-list").owlCarousel({
        items: 3,
        margin: 20,
        loop: true,
        autoplay: true,
        autoplayTimeout: 5000,
        nav: true,
    });
});


// script.js

const images = document.querySelectorAll('.gallery img');

images.forEach((img) => {
    img.addEventListener('mouseover', () => {
        img.title = `This is ${img.alt}`;
    });
});



// Automatically submit the form after files are selected
document.getElementById("uploadButton").addEventListener("click", function () {
    document.getElementById("fileInput").click();
  });

  // Handle file selection and display the file name
  document.getElementById("fileInput").addEventListener("change", function (event) {
    const files = event.target.files;
    if (files.length > 0) {
      const fileName = files[0].name;
      document.getElementById("fileName").textContent = `File uploaded: ${fileName}`;
      document.getElementById("submitBtn").style.display = "inline-block"; // Show the submit button
    }
  });