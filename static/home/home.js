var text = "Test yourself anytime and from anywhere !!!";
var index = 0;
var speed = 100;

function type() {
  document.getElementById("typing-effect").innerHTML += text.charAt(index);
  index++;
  if (index < text.length) {
    setTimeout(type, speed);
  } else {
    setTimeout(deleteText, speed/2);
  }
}

function deleteText() {
  var currentText = document.getElementById("typing-effect").innerHTML;
  if (currentText.length > 0) {
    document.getElementById("typing-effect").innerHTML = currentText.slice(0, -1);
    setTimeout(deleteText, speed);
  } else {
    index = 0;
    setTimeout(type, speed*2);
  }
}

type();



const navbarLinks = document.querySelectorAll('nav a');

navbarLinks.forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    const target = document.querySelector(link.getAttribute('href'));
    target.scrollIntoView({ behavior: "smooth"});
  });
});




const slides = document.querySelectorAll(".carousel-slide");
const prevBtn = document.querySelector(".prev");
const nextBtn = document.querySelector(".next");

let currentSlide = 0;

function showSlide(index) {
  if (index < 0) {
    currentSlide = slides.length - 1;
  } else if (index >= slides.length) {
    currentSlide = 0;
  }
  slides.forEach((slide) => slide.classList.remove("active"));
  slides[currentSlide].classList.add("active");
}

setInterval(() => {
  nextSlide()
}, 3000);



function nextSlide() {
  currentSlide++;
  showSlide(currentSlide);
}

showSlide(currentSlide);
