document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const brushSize = document.getElementById('brushSize');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    const predictionResult = document.getElementById('prediction');
    const confidenceBar = document.querySelector('.confidence-bar');
    const predictionHistory = document.getElementById('predictionHistory');

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Set initial canvas state
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = brushSize.value;

    // Drawing functions
    function draw(e) {
        if (!isDrawing) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        
        [lastX, lastY] = [x, y];
    }

    function startDrawing(e) {
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        [lastX, lastY] = [e.clientX - rect.left, e.clientY - rect.top];
    }

    // Event listeners for drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', () => isDrawing = false);
    canvas.addEventListener('mouseout', () => isDrawing = false);

    // Touch event listeners for mobile
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    });

    canvas.addEventListener('touchend', () => {
        const mouseEvent = new MouseEvent('mouseup');
        canvas.dispatchEvent(mouseEvent);
    });

    // Brush size control
    brushSize.addEventListener('input', (e) => {
        ctx.lineWidth = e.target.value;
    });

    // Clear canvas
    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'black';
        predictionResult.textContent = '';
        confidenceBar.style.width = '0%';
    });

    // Make prediction
    predictBtn.addEventListener('click', async () => {
        try {
            const imageData = canvas.toDataURL('image/png');
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const result = await response.json();
            
            // Update prediction display
            predictionResult.textContent = `Predicted Digit: ${result.prediction}`;
            predictionResult.classList.add('show');
            
            // Update confidence bar
            const confidencePercent = (result.confidence * 100).toFixed(1);
            confidenceBar.style.width = `${confidencePercent}%`;
            
            // Add to history
            const historyItem = document.createElement('li');
            historyItem.innerHTML = `
                <span>Digit ${result.prediction}</span>
                <span>${confidencePercent}% confidence</span>
            `;
            predictionHistory.insertBefore(historyItem, predictionHistory.firstChild);
            
            // Keep only last 5 predictions
            if (predictionHistory.children.length > 5) {
                predictionHistory.removeChild(predictionHistory.lastChild);
            }
            
        } catch (error) {
            console.error('Error:', error);
            predictionResult.textContent = 'Error making prediction';
            predictionResult.style.color = 'red';
        }
    });
}); 