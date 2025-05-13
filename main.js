/* nav bar toggle */

let menuIcon = document.querySelector('#menu-icon');
let navbar = document.querySelector ('.navbar');


menuIcon.onclick = () => {
    menuIcon.classList.toggle('fa-xmark');
    navbar.classList.toggle('active');
};





/* transparent scrolling */
const header = document.querySelector('.header');

window.addEventListener('scroll', () => {
if (window.scrollY > 50) {
    header.classList.add('header-scrolled')
} else if(window.scrollY <= 50){
    header.classList.remove('header-scrolled');
}
})




/* scroll section active link */

let sections = document.querySelectorAll('section'); 
let navLinks = document.querySelectorAll('header nav a');

window.onscroll = () => {
    sections.forEach(sec => {
        let top = window.scrollY;
        let offset = sec.offsetTop - 150;
        let height = sec.offsetHeight;
        let id = sec.getAttribute('id');

        
        if(top >= offset && top < offset + height){
            navLinks.forEach(links => 
                {
                    links.classList.remove('active');

                    // Fix the selector syntax
                    try {
                        document.querySelector('header nav a[href="#' + id + '"]').classList.add('active');
                    } catch(e) {
                        console.log('Could not find selector for: ' + id);
                    }
            });
        };
    });
    /* sticky navbar */
    let header = document.querySelector('header');
    header.classList.toggle('sticky', window.scrollY > 100);


/* remove toggle icon and navbar */
menuIcon.classList.remove('fa-xmark');
navbar.classList.remove('active');
};


/* scroll reveal */

window.sr = ScrollReveal();

ScrollReveal({
   distance: '80px',
   duration: 2000,
   delay: 200,
});

ScrollReveal().reveal('.home-content, heading', {origin: 'top'});
ScrollReveal().reveal('.home-img, .services-container, .portfolio-box, .contact form', {origin: 'bottom'});
ScrollReveal().reveal('.home-content h1, .about-img', {origin:'left'});
ScrollReveal().reveal('.home-contact p, about-content', {origin: 'right'});
ScrollReveal().reveal('.skills', {origin:'right'});
ScrollReveal().reveal('')


/*sign up/ sign in pop up*/

// Get the elements
const signUpBox = document.getElementById('signUpBox');
const signInBox = document.getElementById('signInBox');
const toggleToSignUp = document.getElementById('toggleToSignUp');
const toggleToSignIn = document.getElementById('toggleToSignIn');

// Initially show the Sign Up box and hide the Sign In box
signUpBox.style.display = 'block'; // Show the sign up form initially
signInBox.style.display = 'none'; // Hide the sign in form initially

// When the user clicks on "Don't have an account? Sign Up", show the sign up form
toggleToSignUp.addEventListener('click', () => {
  signUpBox.style.display = 'block';
  signInBox.style.display = 'none'; // Hide the sign in form
});

// When the user clicks on "Already have an account? Sign In", show the sign in form
toggleToSignIn.addEventListener('click', () => {
  signInBox.style.display = 'block';
  signUpBox.style.display = 'none'; // Hide the sign up form
});




/* typed js */


const typed = new Typed('.multiple-text', 
    {
    strings: ['Bohol', 'Boracay', 'El Nido'],
    typeSpeed: 40,
    backSpeed: 20,
    backDelay: 3000,
    startDelay: 0,
    loop: true,
});


// Get all the slides and total number of slides
const slides = document.querySelectorAll('.slide');
const totalSlides = slides.length;
let currentSlideIndex = 0;

// Function to set the background image of the section with a sliding effect
function setBackgroundImage(index) {
    const homeSection = document.querySelector('.home');
    const bgImage = slides[index].getAttribute('data-bg'); // Get the background image from the data attribute
    
    // Triggering the background image change with animation
    homeSection.style.backgroundImage = `url(${bgImage})`; // Set the background image of the section

    // Apply the fade from left to right animation
    homeSection.style.animation = 'fadeFromLeft 1s ease-in-out'; // Trigger the slide animation
}

// Function to move to the next slide
function nextSlide() {
    currentSlideIndex = (currentSlideIndex + 1) % totalSlides; // Loop back to the first slide
    setBackgroundImage(currentSlideIndex);
}

// Function to start the slideshow
function startSlideshow() {
    // Initially set the background image of the first slide
    setBackgroundImage(currentSlideIndex);

    // Change to the next slide every 3500ms (3.5 seconds)
    setInterval(nextSlide, 3500); // Change slide every 3500ms
}

// Start the slideshow when the page loads
window.onload = startSlideshow;


