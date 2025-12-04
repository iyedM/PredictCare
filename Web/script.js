        // Toggle switches functionality
        document.querySelectorAll('.toggle-group').forEach(toggle => {
            toggle.addEventListener('click', function() {
                this.classList.toggle('active');
                const input = this.querySelector('input[type="hidden"]');
                input.value = this.classList.contains('active') ? '1' : '0';
            });
        });

        // Form submission
        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const submitBtn = this.querySelector('.submit-btn');
            const emptyState = document.getElementById('emptyState');
            const resultDisplay = document.getElementById('resultDisplay');
            const errorMessage = document.getElementById('errorMessage');

            // Reset states
            errorMessage.classList.remove('visible');
            submitBtn.classList.add('loading');
            submitBtn.disabled = true;

            // Collect form data
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => {
                data[key] = parseFloat(value);
            });

            try {
                const response = await fetch('http://127.0.0.1:5000/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify([data])
                });

                const result = await response.json();

                if (result.error) {
                    throw new Error(result.error);
                }

                // Show results
                emptyState.style.display = 'none';
                resultDisplay.classList.add('visible');

                const probability = result[0].probability;
                const prediction = result[0].prediction;

                // Animate probability display
                animateProbability(probability);
                updateRiskLevel(probability);
                updateGauge(probability);

            } catch (err) {
                console.error(err);
                errorMessage.querySelector('p').textContent = 
                    err.message || 'Unable to connect to the prediction service. Please ensure the Flask API is running on http://127.0.0.1:5000';
                errorMessage.classList.add('visible');
                emptyState.style.display = 'none';
                resultDisplay.classList.remove('visible');
            } finally {
                submitBtn.classList.remove('loading');
                submitBtn.disabled = false;
            }
        });

        function animateProbability(targetProb) {
            const display = document.getElementById('probabilityValue');
            const targetPercent = Math.round(targetProb * 100);
            let current = 0;
            const duration = 1000;
            const increment = targetPercent / (duration / 16);

            function update() {
                current += increment;
                if (current >= targetPercent) {
                    display.textContent = targetPercent + '%';
                    return;
                }
                display.textContent = Math.round(current) + '%';
                requestAnimationFrame(update);
            }
            update();
        }

        function updateRiskLevel(probability) {
            const badge = document.getElementById('riskBadge');
            const level = document.getElementById('riskLevel');
            const description = document.getElementById('riskDescription');

            badge.classList.remove('low', 'medium', 'high');

            if (probability < 0.3) {
                badge.classList.add('low');
                level.textContent = 'Low Risk';
                description.textContent = 'Great news! Based on the provided health indicators, you have a lower risk of developing diabetes. Continue maintaining a healthy lifestyle with regular physical activity and a balanced diet.';
            } else if (probability < 0.6) {
                badge.classList.add('medium');
                level.textContent = 'Moderate Risk';
                description.textContent = 'You have a moderate risk of developing diabetes. Consider consulting with a healthcare provider about preventive measures. Focus on increasing physical activity, maintaining a healthy weight, and monitoring your blood sugar levels.';
            } else {
                badge.classList.add('high');
                level.textContent = 'High Risk';
                description.textContent = 'Based on the health indicators provided, you may have an elevated risk for diabetes. We strongly recommend consulting with a healthcare professional for a comprehensive evaluation and personalized health plan.';
            }

            // Update badge icon based on risk
            const icon = badge.querySelector('svg');
            if (probability < 0.3) {
                icon.innerHTML = '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>';
            } else if (probability < 0.6) {
                icon.innerHTML = '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>';
            } else {
                icon.innerHTML = '<path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>';
            }
        }

        function updateGauge(probability) {
            const needle = document.getElementById('gaugeNeedle');
            const gaugePath = document.getElementById('gaugePath');
            
            // Needle rotation: -90deg (0%) to 90deg (100%)
            const rotation = -90 + (probability * 180);
            needle.style.transform = `translateX(-50%) rotate(${rotation}deg)`;

            // Gauge fill: total path length is 251
            const fillAmount = 251 - (probability * 251);
            gaugePath.style.strokeDashoffset = fillAmount;
        }