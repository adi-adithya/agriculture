// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function () {
    // Initialize Bootstrap tooltips and popovers
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize dropdown functionality for navigation
    var dropdownElementList = [].slice.call(document.querySelectorAll('.dropdown-toggle'));
    var dropdownList = dropdownElementList.map(function (dropdownToggleEl) {
        return new bootstrap.Dropdown(dropdownToggleEl);
    });

    // Handle click on AgriPrice Tracker logo/brand to restart the site
    const brandLogo = document.querySelector('.navbar-brand');
    if (brandLogo) {
        brandLogo.addEventListener('click', function (e) {
            e.preventDefault();
            restartSite();
        });
    }

    // Function to restart the site
    function restartSite() {
        // Hide results containers if they're visible
        if (document.getElementById('resultsContainer')) {
            document.getElementById('resultsContainer').style.display = 'none';
        }
        if (document.getElementById('noResultsContainer')) {
            document.getElementById('noResultsContainer').style.display = 'none';
        }

        // Show welcome container
        if (document.getElementById('welcomeContainer')) {
            document.getElementById('welcomeContainer').style.display = 'block';
        }

        // Reset search form if it exists
        const searchForm = document.getElementById('searchForm');
        if (searchForm) {
            searchForm.reset();
        }

        // Reset results title and content
        if (document.getElementById('resultsTitle')) {
            document.getElementById('resultsTitle').textContent = 'Search Results';
        }
        if (document.getElementById('resultsContent')) {
            document.getElementById('resultsContent').innerHTML = '';
        }

        // Scroll to top of page
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Function to handle category search from navigation
    window.searchCategory = function (category) {
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.value = category;
            // If there's a form submit event handler, trigger it
            const searchForm = document.getElementById('searchForm');
            if (searchForm) {
                // Create and dispatch submit event
                const submitEvent = new Event('submit', {
                    bubbles: true,
                    cancelable: true
                });
                searchForm.dispatchEvent(submitEvent);
            }
        }

        // Alternatively, if form submission is handled differently:
        // Show loading spinner
        const loadingSpinner = document.getElementById('loadingSpinner');
        if (loadingSpinner) {
            loadingSpinner.style.display = 'flex';
        }

        // Simulate loading delay for demonstration purposes
        setTimeout(function () {
            // Hide loading spinner
            if (loadingSpinner) {
                loadingSpinner.style.display = 'none';
            }

            // Here you would normally process the results
            // For now, we'll just show a placeholder message
            if (document.getElementById('welcomeContainer')) {
                document.getElementById('welcomeContainer').style.display = 'none';
            }

            if (document.getElementById('resultsContainer')) {
                document.getElementById('resultsContainer').style.display = 'block';
                document.getElementById('resultsTitle').textContent = category + ' Price Data';
                document.getElementById('resultsInfo').innerHTML =
                    '<i class="fas fa-info-circle"></i> Showing current prices and trends for ' + category + '.';
            }
        }, 1000);
    };

    // Handle quick search buttons
    window.quickSearch = function (product) {
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.value = product;
            // If there's a form submit event handler, trigger it
            const searchForm = document.getElementById('searchForm');
            if (searchForm) {
                // Create and dispatch submit event
                const submitEvent = new Event('submit', {
                    bubbles: true,
                    cancelable: true
                });
                searchForm.dispatchEvent(submitEvent);
            }
        }
    };

    // Team showcase interactions - adding hover effects
    const teamMembers = document.querySelectorAll('.team-member');
    teamMembers.forEach(member => {
        member.addEventListener('mouseenter', function () {
            this.classList.add('team-member-hover');
        });

        member.addEventListener('mouseleave', function () {
            this.classList.remove('team-member-hover');
        });
    });

    // Handle search form submission
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', function (e) {
            e.preventDefault(); // Prevent actual form submission

            const query = document.getElementById('searchInput').value;
            const location = document.getElementById('locationInput').value;
            const period = document.getElementById('periodInput').value;

            // Show loading spinner
            const loadingSpinner = document.getElementById('loadingSpinner');
            if (loadingSpinner) {
                loadingSpinner.style.display = 'flex';
            }

            // Here you would normally fetch data from your backend
            // For now, we'll simulate a request with timeout
            setTimeout(function () {
                // Hide loading spinner
                if (loadingSpinner) {
                    loadingSpinner.style.display = 'none';
                }

                // Hide welcome container
                if (document.getElementById('welcomeContainer')) {
                    document.getElementById('welcomeContainer').style.display = 'none';
                }

                // Show results (for demo, always show some results)
                if (document.getElementById('resultsContainer')) {
                    document.getElementById('resultsContainer').style.display = 'block';
                    document.getElementById('resultsTitle').textContent = 'Results for: ' + query;
                }
            }, 1500);
        });
    }

    // Handle geolocation button
    const getLocationBtn = document.getElementById('getLocationBtn');
    if (getLocationBtn) {
        getLocationBtn.addEventListener('click', function () {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    function (position) {
                        const locationInput = document.getElementById('locationInput');
                        if (locationInput) {
                            locationInput.value = position.coords.latitude + ',' + position.coords.longitude;
                        }
                    },
                    function (error) {
                        console.error("Error getting location:", error);
                        alert("Unable to retrieve your location. Please enter it manually.");
                    }
                );
            } else {
                alert("Geolocation is not supported by this browser.");
            }
        });
    }

    // Reset search function
    window.resetSearch = function () {
        restartSite();
    };

    // Add animated hover effect for navbar items
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('mouseenter', function () {
            this.classList.add('nav-link-hover');
        });

        link.addEventListener('mouseleave', function () {
            this.classList.remove('nav-link-hover');
        });
    });

    // Pulse animation for leaf icon in navbar brand
    const leafIcon = document.querySelector('.navbar-brand .fa-leaf');
    if (leafIcon) {
        // The animation is handled by CSS, we're just ensuring the class is there
        if (!leafIcon.classList.contains('animate-pulse')) {
            leafIcon.classList.add('animate-pulse');
        }
    }
});