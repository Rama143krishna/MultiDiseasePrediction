// =======================================================
// Multi Disease Prediction System - Premium JavaScript UI
// =======================================================

let diabetesChart = null;
let cardioChart = null;

// ======================================
// Utility Functions
// ======================================
function showLoading(elementId, text = "Analyzing...") {
    const el = document.getElementById(elementId);
    el.innerHTML = `<span class="loader"></span> ${text}`;
}

function showResult(elementId, message, success = true) {
    const el = document.getElementById(elementId);
    el.innerHTML = message;

    el.style.transform = "scale(0.95)";
    setTimeout(() => {
        el.style.transform = "scale(1)";
    }, 150);

    if (success) {
        el.classList.add("success-glow");
        setTimeout(() => el.classList.remove("success-glow"), 1200);
    }
}

function validateNumber(id, name) {
    const value = document.getElementById(id).value.trim();
    if (value === "" || isNaN(value)) {
        throw `${name} is required`;
    }
    return value;
}

function animateCounter(elementId, targetValue) {
    const el = document.getElementById(elementId);
    let start = 0;
    const duration = 1200;
    const step = targetValue / (duration / 16);

    const timer = setInterval(() => {
        start += step;
        if (start >= targetValue) {
            start = targetValue;
            clearInterval(timer);
        }
        el.innerHTML = start.toFixed(1) + "%";
    }, 16);
}

// ======================================
// Toast Notification
// ======================================
function showToast(msg, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerText = msg;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add("show");
    }, 100);

    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 500);
    }, 3000);
}

// ======================================
// Diabetes Prediction
// ======================================
async function predictDiabetes() {
    try {
        showLoading("diabetesResult");

        const payload = {
            gender: document.getElementById("d_gender").value,
            age: validateNumber("d_age", "Age"),
            hypertension: validateNumber("d_hyper", "Hypertension"),
            heart_disease: validateNumber("d_heart", "Heart Disease"),
            smoking: document.getElementById("d_smoking").value,
            bmi: validateNumber("d_bmi", "BMI"),
            hba1c: validateNumber("d_hba1c", "HbA1c"),
            glucose: validateNumber("d_glucose", "Glucose")
        };

        const response = await fetch("/predict_diabetes", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.status === "success") {
            const prob = (data.probability * 100).toFixed(1);

            showResult(
                "diabetesResult",
                `🩸 <b>${data.prediction}</b><br>Risk Probability: <span class="percent">${prob}%</span>`
            );

            drawDiabetesChart(prob);
            showToast("Diabetes prediction completed!", "success");
        } else {
            throw data.message;
        }

    } catch (error) {
        showResult("diabetesResult", `❌ ${error}`, false);
        showToast(error, "error");
    }
}

// ======================================
// Cardio Prediction
// ======================================
async function predictCardio() {
    try {
        showLoading("cardioResult");

        const payload = {
            age: validateNumber("c_age", "Age"),
            gender: validateNumber("c_gender", "Gender"),
            height: validateNumber("c_height", "Height"),
            weight: validateNumber("c_weight", "Weight"),
            ap_hi: validateNumber("c_ap_hi", "Systolic BP"),
            ap_lo: validateNumber("c_ap_lo", "Diastolic BP"),
            cholesterol: validateNumber("c_chol", "Cholesterol"),
            gluc: validateNumber("c_gluc", "Glucose"),
            smoke: validateNumber("c_smoke", "Smoke"),
            alco: validateNumber("c_alco", "Alcohol"),
            active: validateNumber("c_active", "Activity")
        };

        const response = await fetch("/predict_cardio", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.status === "success") {
            const prob = (data.probability * 100).toFixed(1);

            showResult(
                "cardioResult",
                `❤️ <b>${data.prediction}</b><br>Risk Probability: <span class="percent">${prob}%</span>`
            );

            drawCardioChart(prob);
            showToast("Cardio prediction completed!", "success");
        } else {
            throw data.message;
        }

    } catch (error) {
        showResult("cardioResult", `❌ ${error}`, false);
        showToast(error, "error");
    }
}

// ======================================
// Charts
// ======================================
function drawDiabetesChart(prob) {
    const ctx = document.getElementById("diabetesChart").getContext("2d");

    if (diabetesChart) diabetesChart.destroy();

    diabetesChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Risk", "Safe"],
            datasets: [{
                data: [prob, 100 - prob],
                backgroundColor: [
                    "rgba(255, 99, 132, 0.9)",
                    "rgba(0, 255, 170, 0.7)"
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        color: "#fff",
                        font: {size: 14}
                    }
                }
            }
        }
    });
}

function drawCardioChart(prob) {
    const ctx = document.getElementById("cardioChart").getContext("2d");

    if (cardioChart) cardioChart.destroy();

    cardioChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Risk %"],
            datasets: [{
                label: "Cardio Risk",
                data: [prob],
                backgroundColor: [
                    "rgba(255, 0, 90, 0.9)"
                ],
                borderRadius: 12
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: "#fff"
                    }
                },
                x: {
                    ticks: {
                        color: "#fff"
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: "#fff"
                    }
                }
            }
        }
    });
}

// ======================================
// Input Glow Effect
// ======================================
document.addEventListener("DOMContentLoaded", () => {
    const inputs = document.querySelectorAll("input");

    inputs.forEach(input => {
        input.addEventListener("focus", () => {
            input.style.boxShadow = "0 0 15px rgba(0,255,255,0.8)";
        });

        input.addEventListener("blur", () => {
            input.style.boxShadow = "none";
        });
    });

    showToast("System Ready 🚀", "info");
});