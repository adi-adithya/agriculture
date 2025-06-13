$(document).ready(function () {
    // Handle form submission
    $("#searchForm").submit(function (e) {
        e.preventDefault();
        const query = $("#searchInput").val().trim();
        if (query) {
            performSearch();
        }
    });

    // Get location button handler
    $("#getLocationBtn").click(function () {
        if (navigator.geolocation) {
            showLoading("Getting your location...");
            navigator.geolocation.getCurrentPosition(
                function (position) {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    $("#locationInput").val(`${lat.toFixed(4)},${lon.toFixed(4)}`);
                    hideLoading();
                },
                function (error) {
                    hideLoading();
                    alert("Unable to get your location. Please enter manually.");
                    console.error("Geolocation error:", error);
                }
            );
        } else {
            alert("Geolocation is not supported by your browser.");
        }
    });

    // Handle search suggestions
    $("#searchInput").on("input", function () {
        const input = $(this).val().toLowerCase();
        if (input.length > 0) {
            const suggestions = $("#productSuggestions option").map(function () {
                return $(this).val();
            }).get();

            const filtered = suggestions.filter(item =>
                item.toLowerCase().includes(input)
            ).slice(0, 5);

            if (filtered.length > 0) {
                updateSuggestions(filtered);
                $("#searchSuggestions").show();
            } else {
                $("#searchSuggestions").hide();
            }
        } else {
            $("#searchSuggestions").hide();
        }
    });

    // Hide suggestions when clicking outside
    $(document).on("click", function (e) {
        if (!$(e.target).closest("#searchInput, #searchSuggestions").length) {
            $("#searchSuggestions").hide();
        }
    });
});

function updateSuggestions(items) {
    const container = $("#searchSuggestions");
    container.empty();

    items.forEach(item => {
        const element = $("<div>").addClass("datalist-item").text(item);
        element.on("click", function () {
            $("#searchInput").val(item);
            $("#searchSuggestions").hide();
        });
        container.append(element);
    });
}

function quickSearch(query) {
    $("#searchInput").val(query);
    performSearch();
}

function searchCategory(category) {
    $("#searchInput").val(category);
    performSearch();
}

function performSearch() {
    const query = $("#searchInput").val().trim();
    const location = $("#locationInput").val().trim();
    const period = $("#periodInput").val();

    if (!query) return;

    showLoading();

    $.ajax({
        url: "/search",
        method: "POST",
        data: {
            query: query,
            location: location,
            period: period
        },
        success: function (response) {
            hideLoading();

            if (response.status === "success") {
                displayResults(response);
            } else {
                showNoResults(response.message);
            }
        },
        error: function (xhr, status, error) {
            hideLoading();
            showNoResults("An error occurred while fetching data. Please try again.");
            console.error("Search error:", error);
        }
    });
}

function displayResults(data) {
    // Hide welcome content and show results
    $("#welcomeContainer").hide();
    $("#noResultsContainer").hide();
    $("#resultsContainer").show();

    // Update results title and info
    $("#resultsTitle").text(`Search Results for "${data.query}"`);
    $("#resultsInfo").html(`<i class="fas fa-info-circle"></i> Showing ${data.results.length} products for the last ${data.period} days.`);

    // Clear previous results
    $("#resultsContent").empty();

    // Add new results
    data.results.forEach(function (item) {
        const template = document.getElementById("productCardTemplate");
        const clone = document.importNode(template.content, true);

        // Clone will have unique IDs for each tab
        const uniqueId = `product-${Math.random().toString(36).substr(2, 9)}`;

        // Update tab IDs to be unique
        clone.querySelector("[data-bs-target='#tab-history']").setAttribute("data-bs-target", `#tab-history-${uniqueId}`);
        clone.querySelector("[data-bs-target='#tab-markets']").setAttribute("data-bs-target", `#tab-markets-${uniqueId}`);
        clone.querySelector("[data-bs-target='#tab-prediction']").setAttribute("data-bs-target", `#tab-prediction-${uniqueId}`);

        clone.querySelector("#tab-history").id = `tab-history-${uniqueId}`;
        clone.querySelector("#tab-markets").id = `tab-markets-${uniqueId}`;
        clone.querySelector("#tab-prediction").id = `tab-prediction-${uniqueId}`;

        // Set content
        clone.querySelector(".product-title").textContent = item.product;
        clone.querySelector(".current-price").textContent = `₹${item.current_price.toFixed(2)}`;
        clone.querySelector(".product-category").textContent = item.category;
        clone.querySelector(".product-market").textContent = item.market;
        clone.querySelector(".price-date").textContent = item.date;

        // Set trend
        const trendElem = clone.querySelector(".price-trend");
        trendElem.textContent = item.trend;
        if (item.trend.includes("Rise") || item.trend.includes("Rising")) {
            trendElem.classList.add("trend-rising");
        } else if (item.trend.includes("Fall") || item.trend.includes("Falling")) {
            trendElem.classList.add("trend-falling");
        } else {
            trendElem.classList.add("trend-stable");
        }

        // Set trend description
        clone.querySelector(".trend-description").textContent = item.trend_description;

        // Set charts
        if (item.history_plot) {
            clone.querySelector(".price-history-chart").src = item.history_plot + "?t=" + new Date().getTime();
        }

        if (item.market_comparison_plot) {
            clone.querySelector(".market-comparison-chart").src = item.market_comparison_plot + "?t=" + new Date().getTime();
        } else {
            clone.querySelector("[data-bs-target='#tab-markets-" + uniqueId + "']").parentNode.style.display = "none";
        }

        if (item.prediction_plot) {
            clone.querySelector(".prediction-chart").src = item.prediction_plot + "?t=" + new Date().getTime();

            // Add prediction summary
            const predSummary = clone.querySelector(".prediction-summary");
            if (item.predicted_prices && item.predicted_prices.length > 0) {
                const lastPrediction = item.predicted_prices[item.predicted_prices.length - 1];
                const firstPrediction = item.predicted_prices[0];
                const priceDiff = lastPrediction.predicted_price - item.current_price;
                const percentChange = (priceDiff / item.current_price) * 100;

                let predictionText = `Predicted price after 7 days: <strong>₹${lastPrediction.predicted_price.toFixed(2)}</strong>`;
                predictionText += ` (${percentChange > 0 ? '+' : ''}${percentChange.toFixed(2)}%)`;

                const predBadge = document.createElement('div');
                predBadge.className = 'prediction-badge';
                predBadge.innerHTML = predictionText;
                predSummary.appendChild(predBadge);
            }
        } else {
            clone.querySelector("[data-bs-target='#tab-prediction-" + uniqueId + "']").parentNode.style.display = "none";
        }

        // Set factors
        const factorsContainer = clone.querySelector(".factors-list");
        if (item.factors && item.factors.length > 0) {
            item.factors.forEach(function (factor) {
                const factorElem = document.createElement('div');
                factorElem.className = `factor-item factor-${factor.importance}`;
                factorElem.innerHTML = `<strong>${factor.factor}:</strong> ${factor.impact}`;
                factorsContainer.appendChild(factorElem);
            });
        } else {
            factorsContainer.innerHTML = '<div class="text-muted">No significant factors identified.</div>';
        }

        $("#resultsContent").append(clone);
    });

    // Initialize Bootstrap tabs
    const triggerTabList = [].slice.call(document.querySelectorAll('.nav-tabs button'));
    triggerTabList.forEach(function (triggerEl) {
        const tabTrigger = new bootstrap.Tab(triggerEl);
        triggerEl.addEventListener('click', function (event) {
            event.preventDefault();
            tabTrigger.show();
        });
    });
}

function showNoResults(message) {
    $("#welcomeContainer").hide();
    $("#resultsContainer").hide();
    $("#noResultsContainer").show();
    $("#noResultsMessage").text(message || "No results found. Please try a different search term.");
}

function resetSearch() {
    $("#searchInput").val("");
    $("#noResultsContainer").hide();
    $("#resultsContainer").hide();
    $("#welcomeContainer").show();
}

function showLoading(message) {
    $("#loadingSpinner").find(".spinner-text").text(message || "Loading price data...");
    $("#loadingSpinner").show();
}

function hideLoading() {
    $("#loadingSpinner").hide();
} $(document).ready(function () {
    // Handle form submission
    $("#searchForm").submit(function (e) {
        e.preventDefault();
        const query = $("#searchInput").val().trim();
        if (query) {
            performSearch();
        }
    });

    // Get location button handler
    $("#getLocationBtn").click(function () {
        if (navigator.geolocation) {
            showLoading("Getting your location...");
            navigator.geolocation.getCurrentPosition(
                function (position) {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    $("#locationInput").val(`${lat.toFixed(4)},${lon.toFixed(4)}`);
                    hideLoading();
                },
                function (error) {
                    hideLoading();
                    alert("Unable to get your location. Please enter manually.");
                    console.error("Geolocation error:", error);
                }
            );
        } else {
            alert("Geolocation is not supported by your browser.");
        }
    });

    // Handle search suggestions
    $("#searchInput").on("input", function () {
        const input = $(this).val().toLowerCase();
        if (input.length > 0) {
            const suggestions = $("#productSuggestions option").map(function () {
                return $(this).val();
            }).get();

            const filtered = suggestions.filter(item =>
                item.toLowerCase().includes(input)
            ).slice(0, 5);

            if (filtered.length > 0) {
                updateSuggestions(filtered);
                $("#searchSuggestions").show();
            } else {
                $("#searchSuggestions").hide();
            }
        } else {
            $("#searchSuggestions").hide();
        }
    });

    // Hide suggestions when clicking outside
    $(document).on("click", function (e) {
        if (!$(e.target).closest("#searchInput, #searchSuggestions").length) {
            $("#searchSuggestions").hide();
        }
    });
});

function updateSuggestions(items) {
    const container = $("#searchSuggestions");
    container.empty();

    items.forEach(item => {
        const element = $("<div>").addClass("datalist-item").text(item);
        element.on("click", function () {
            $("#searchInput").val(item);
            $("#searchSuggestions").hide();
        });
        container.append(element);
    });
}

function quickSearch(query) {
    $("#searchInput").val(query);
    performSearch();
}

function searchCategory(category) {
    $("#searchInput").val(category);
    performSearch();
}

function performSearch() {
    const query = $("#searchInput").val().trim();
    const location = $("#locationInput").val().trim();
    const period = $("#periodInput").val();

    if (!query) return;

    showLoading();

    $.ajax({
        url: "/search",
        method: "POST",
        data: {
            query: query,
            location: location,
            period: period
        },
        success: function (response) {
            hideLoading();

            if (response.status === "success") {
                displayResults(response);
            } else {
                showNoResults(response.message);
            }
        },
        error: function (xhr, status, error) {
            hideLoading();
            showNoResults("An error occurred while fetching data. Please try again.");
            console.error("Search error:", error);
        }
    });
}

function displayResults(data) {
    // Hide welcome content and show results
    $("#welcomeContainer").hide();
    $("#noResultsContainer").hide();
    $("#resultsContainer").show();

    // Update results title and info
    $("#resultsTitle").text(`Search Results for "${data.query}"`);
    $("#resultsInfo").html(`<i class="fas fa-info-circle"></i> Showing ${data.results.length} products for the last ${data.period} days.`);

    // Clear previous results
    $("#resultsContent").empty();

    // Add new results
    data.results.forEach(function (item) {
        const template = document.getElementById("productCardTemplate");
        const clone = document.importNode(template.content, true);

        // Clone will have unique IDs for each tab
        const uniqueId = `product-${Math.random().toString(36).substr(2, 9)}`;

        // Update tab IDs to be unique
        clone.querySelector("[data-bs-target='#tab-history']").setAttribute("data-bs-target", `#tab-history-${uniqueId}`);
        clone.querySelector("[data-bs-target='#tab-markets']").setAttribute("data-bs-target", `#tab-markets-${uniqueId}`);
        clone.querySelector("[data-bs-target='#tab-prediction']").setAttribute("data-bs-target", `#tab-prediction-${uniqueId}`);

        clone.querySelector("#tab-history").id = `tab-history-${uniqueId}`;
        clone.querySelector("#tab-markets").id = `tab-markets-${uniqueId}`;
        clone.querySelector("#tab-prediction").id = `tab-prediction-${uniqueId}`;

        // Set content
        clone.querySelector(".product-title").textContent = item.product;
        clone.querySelector(".current-price").textContent = `₹${item.current_price.toFixed(2)}`;
        clone.querySelector(".product-category").textContent = item.category;
        clone.querySelector(".product-market").textContent = item.market;
        clone.querySelector(".price-date").textContent = item.date;

        // Set trend
        const trendElem = clone.querySelector(".price-trend");
        trendElem.textContent = item.trend;
        if (item.trend.includes("Rise") || item.trend.includes("Rising")) {
            trendElem.classList.add("trend-rising");
        } else if (item.trend.includes("Fall") || item.trend.includes("Falling")) {
            trendElem.classList.add("trend-falling");
        } else {
            trendElem.classList.add("trend-stable");
        }

        // Set trend description
        clone.querySelector(".trend-description").textContent = item.trend_description;

        // Set charts
        if (item.history_plot) {
            clone.querySelector(".price-history-chart").src = item.history_plot + "?t=" + new Date().getTime();
        }

        if (item.market_comparison_plot) {
            clone.querySelector(".market-comparison-chart").src = item.market_comparison_plot + "?t=" + new Date().getTime();
        } else {
            clone.querySelector("[data-bs-target='#tab-markets-" + uniqueId + "']").parentNode.style.display = "none";
        }

        if (item.prediction_plot) {
            clone.querySelector(".prediction-chart").src = item.prediction_plot + "?t=" + new Date().getTime();

            // Add prediction summary
            const predSummary = clone.querySelector(".prediction-summary");
            if (item.predicted_prices && item.predicted_prices.length > 0) {
                const lastPrediction = item.predicted_prices[item.predicted_prices.length - 1];
                const firstPrediction = item.predicted_prices[0];
                const priceDiff = lastPrediction.predicted_price - item.current_price;
                const percentChange = (priceDiff / item.current_price) * 100;

                let predictionText = `Predicted price after 7 days: <strong>₹${lastPrediction.predicted_price.toFixed(2)}</strong>`;
                predictionText += ` (${percentChange > 0 ? '+' : ''}${percentChange.toFixed(2)}%)`;

                const predBadge = document.createElement('div');
                predBadge.className = 'prediction-badge';
                predBadge.innerHTML = predictionText;
                predSummary.appendChild(predBadge);
            }
        } else {
            clone.querySelector("[data-bs-target='#tab-prediction-" + uniqueId + "']").parentNode.style.display = "none";
        }

        // Set factors
        const factorsContainer = clone.querySelector(".factors-list");
        if (item.factors && item.factors.length > 0) {
            item.factors.forEach(function (factor) {
                const factorElem = document.createElement('div');
                factorElem.className = `factor-item factor-${factor.importance}`;
                factorElem.innerHTML = `<strong>${factor.factor}:</strong> ${factor.impact}`;
                factorsContainer.appendChild(factorElem);
            });
        } else {
            factorsContainer.innerHTML = '<div class="text-muted">No significant factors identified.</div>';
        }

        $("#resultsContent").append(clone);
    });

    // Initialize Bootstrap tabs
    const triggerTabList = [].slice.call(document.querySelectorAll('.nav-tabs button'));
    triggerTabList.forEach(function (triggerEl) {
        const tabTrigger = new bootstrap.Tab(triggerEl);
        triggerEl.addEventListener('click', function (event) {
            event.preventDefault();
            tabTrigger.show();
        });
    });
}

function showNoResults(message) {
    $("#welcomeContainer").hide();
    $("#resultsContainer").hide();
    $("#noResultsContainer").show();
    $("#noResultsMessage").text(message || "No results found. Please try a different search term.");
}

function resetSearch() {
    $("#searchInput").val("");
    $("#noResultsContainer").hide();
    $("#resultsContainer").hide();
    $("#welcomeContainer").show();
}

function showLoading(message) {
    $("#loadingSpinner").find(".spinner-text").text(message || "Loading price data...");
    $("#loadingSpinner").show();
}

function hideLoading() {
    $("#loadingSpinner").hide();
}