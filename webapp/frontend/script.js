document.addEventListener('DOMContentLoaded', () => {
    const samplesGrid = document.getElementById('samplesGrid');
    const newSamplesBtn = document.getElementById('newSamplesBtn');
    const predictionHistory = document.getElementById('predictionHistory');
    const correctCount = document.getElementById('correctCount');
    const avgConfidence = document.getElementById('avgConfidence');
    const currentPrediction = document.getElementById('currentPrediction');

    let totalTests = 0;
    let totalCorrect = 0;
    let totalConfidence = 0;

    function createDigitDisplay(pixelValues) {
        // Create a canvas element for precise pixel rendering
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style.width = '112px';  // Display at 4x size
        canvas.style.height = '112px';
        canvas.style.imageRendering = 'pixelated';  // Keep pixels sharp when scaled
        
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(28, 28);
        
        // Fill the image data with grayscale values
        for (let i = 0; i < pixelValues.length; i++) {
            const intensity = Math.floor(pixelValues[i] * 255);
            // Each pixel needs 4 values (r,g,b,a)
            imageData.data[i * 4] = intensity;     // r
            imageData.data[i * 4 + 1] = intensity; // g
            imageData.data[i * 4 + 2] = intensity; // b
            imageData.data[i * 4 + 3] = 255;       // a (fully opaque)
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        // Create container with border
        const container = document.createElement('div');
        container.className = 'mnist-display';
        container.appendChild(canvas);
        
        return container;
    }

    function updateStats() {
        correctCount.textContent = `${totalCorrect}/${totalTests}`;
        const avgConf = totalTests > 0 ? (totalConfidence / totalTests * 100).toFixed(1) : 0;
        avgConfidence.textContent = `${avgConf}%`;
    }

    function addToHistory(prediction) {
        const historyItem = document.createElement('li');
        const accuracy = prediction.true_label === prediction.predicted_label ? 'Correct' : 'Incorrect';
        historyItem.innerHTML = `
            <span>${accuracy} (True: ${prediction.true_label}, Pred: ${prediction.predicted_label})</span>
            <span>${(prediction.confidence * 100).toFixed(1)}% confident</span>
        `;
        predictionHistory.insertBefore(historyItem, predictionHistory.firstChild);
        
        // Keep only last 5 entries
        if (predictionHistory.children.length > 5) {
            predictionHistory.removeChild(predictionHistory.lastChild);
        }
    }

    async function predictSample(sampleId, sampleContainer) {
        try {
            const response = await fetch(`/predict/${sampleId}`);
            if (!response.ok) {
                throw new Error('Prediction failed');
            }
            
            const result = await response.json();
            
            // Update current prediction display
            const isCorrect = result.true_label === result.predicted_label;
            currentPrediction.querySelector('.prediction-details').innerHTML = `
                <div class="prediction ${isCorrect ? 'correct' : 'incorrect'}">
                    <div class="prediction-label">True: ${result.true_label}</div>
                    <div class="prediction-label">Predicted: ${result.predicted_label}</div>
                    <div class="confidence">${(result.confidence * 100).toFixed(1)}% confident</div>
                </div>
            `;
            
            // Update statistics
            totalTests++;
            if (isCorrect) totalCorrect++;
            totalConfidence += result.confidence;
            updateStats();
            
            // Add to history
            addToHistory(result);
            
            // Load new samples after a short delay
            setTimeout(loadNewSamples, 1500);
            
        } catch (error) {
            console.error('Error:', error);
            currentPrediction.querySelector('.prediction-details').innerHTML = `
                <div class="error">
                    Failed to make prediction: ${error.message}
                </div>
            `;
        }
    }

    async function loadNewSamples() {
        try {
            samplesGrid.innerHTML = '';
            currentPrediction.querySelector('.prediction-details').textContent = 'Click a digit to see the prediction';
            
            const response = await fetch('/get_test_samples');
            if (!response.ok) {
                throw new Error('Failed to fetch test samples');
            }
            
            const samples = await response.json();
            
            samples.forEach(sample => {
                const sampleContainer = document.createElement('div');
                sampleContainer.className = 'sample-item clickable';
                
                // Create and add the digit display
                const digitDisplay = createDigitDisplay(sample.pixels);
                sampleContainer.appendChild(digitDisplay);
                
                // Add click handler
                sampleContainer.addEventListener('click', () => {
                    // Remove clickable class from all samples
                    document.querySelectorAll('.sample-item').forEach(item => {
                        item.classList.remove('clickable');
                    });
                    // Highlight selected sample
                    sampleContainer.classList.add('selected');
                    // Make prediction
                    predictSample(sample.id, sampleContainer);
                });
                
                samplesGrid.appendChild(sampleContainer);
            });
            
        } catch (error) {
            console.error('Error:', error);
            samplesGrid.innerHTML = `
                <div class="error">
                    Failed to load test samples: ${error.message}
                </div>
            `;
        }
    }

    // Load initial samples
    loadNewSamples();

    // Add click handler for new samples button
    newSamplesBtn.addEventListener('click', loadNewSamples);
});
