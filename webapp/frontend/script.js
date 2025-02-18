// Game state
const gameState = {
    score: 0,
    streak: 0,
    highScore: localStorage.getItem('highScore') || 0,
    gamesPlayed: parseInt(localStorage.getItem('gamesPlayed')) || 0,
    correctPredictions: 0,
    totalPredictions: 0,
    timeMode: false,
    timeLeft: 30,
    timerInterval: null,
    achievements: JSON.parse(localStorage.getItem('achievements')) || [],
    vsMode: {
        active: false,
        currentRound: 0,
        totalRounds: 10,
        currentDigit: null,
        humanScore: 0,
        aiScore: 0,
        humanCorrect: 0,
        aiCorrect: 0,
        humanTotalTime: 0,
        aiTotalTime: 0,
        roundStartTime: 0,
        currentTime: 0
    }
};

// Achievement definitions
const achievements = {
    firstCorrect: { id: 'firstCorrect', icon: '🎯', title: 'First Hit!', description: 'Got your first correct prediction' },
    streak3: { id: 'streak3', icon: '🔥', title: 'Hot Streak', description: 'Got 3 correct predictions in a row' },
    streak5: { id: 'streak5', icon: '🌟', title: 'Star Student', description: 'Got 5 correct predictions in a row' },
    score100: { id: 'score100', icon: '💯', title: 'Century Club', description: 'Reached 100 points' },
    perfectRound: { id: 'perfectRound', icon: '🏆', title: 'Perfect Round', description: 'Got all predictions correct in one round' }
};

// DOM Elements
const elements = {
    samplesGrid: document.getElementById('samplesGrid'),
    newSamplesBtn: document.getElementById('newSamplesBtn'),
    currentPrediction: document.getElementById('currentPrediction'),
    score: document.getElementById('score'),
    streak: document.getElementById('streak'),
    highScore: document.getElementById('highScore'),
    accuracy: document.getElementById('accuracy'),
    gamesPlayed: document.getElementById('gamesPlayed'),
    timeBar: document.getElementById('timeBar'),
    timeProgress: document.querySelector('.time-progress'),
    practiceMode: document.getElementById('practiceMode'),
    timeMode: document.getElementById('timeMode'),
    achievements: document.getElementById('achievements'),
    gameFeedback: document.getElementById('gameFeedback'),
    vsMode: document.getElementById('vsMode'),
    vsModeHeader: document.getElementById('vsModeHeader'),
    vsModeControls: document.getElementById('vsModeControls'),
    startVsBtn: document.getElementById('startVsBtn'),
    vsResults: document.getElementById('vsResults'),
    humanScore: document.getElementById('humanScore'),
    humanAccuracy: document.getElementById('humanAccuracy'),
    humanTime: document.getElementById('humanTime'),
    aiScore: document.getElementById('aiScore'),
    aiAccuracy: document.getElementById('aiAccuracy'),
    aiTime: document.getElementById('aiTime'),
    vsGameContainer: document.getElementById('vsGameContainer'),
    feedbackPopup: document.getElementById('feedbackPopup'),
    roundTimer: document.getElementById('roundTimer'),
    currentRound: document.getElementById('currentRound')
};

// Initialize the game
function initGame() {
    updateStats();
    loadNewSamples();
    setupEventListeners();
    displayAchievements();
}

// Event Listeners
function setupEventListeners() {
    elements.newSamplesBtn.addEventListener('click', loadNewSamples);
    elements.practiceMode.addEventListener('click', () => switchMode('practice'));
    elements.timeMode.addEventListener('click', () => switchMode('time'));
    elements.vsMode.addEventListener('click', () => switchMode('vs'));
    elements.startVsBtn.addEventListener('click', startVsRound);
    
    // Add event listeners for number buttons
    document.querySelectorAll('.number-btn').forEach(btn => {
        btn.addEventListener('click', () => handleHumanGuess(parseInt(btn.dataset.number)));
    });
}

// Switch game mode
function switchMode(mode) {
    if (mode === 'time' && !gameState.timeMode) {
        startTimeMode();
    } else if (mode === 'practice' && gameState.timeMode) {
        endTimeMode();
    } else if (mode === 'vs') {
        endTimeMode();
        startVsMode();
    } else {
        endVsMode();
    }
    
    elements.practiceMode.classList.toggle('active', mode === 'practice');
    elements.timeMode.classList.toggle('active', mode === 'time');
    elements.vsMode.classList.toggle('active', mode === 'vs');
    elements.timeBar.classList.toggle('active', mode === 'time');
}

// Start time challenge mode
function startTimeMode() {
    gameState.timeMode = true;
    gameState.timeLeft = 30;
    gameState.score = 0;
    gameState.streak = 0;
    updateStats();
    loadNewSamples();
    
    elements.timeProgress.style.width = '100%';
    gameState.timerInterval = setInterval(() => {
        gameState.timeLeft--;
        elements.timeProgress.style.width = `${(gameState.timeLeft / 30) * 100}%`;
        
        if (gameState.timeLeft <= 0) {
            endTimeMode();
        }
    }, 1000);
}

// End time challenge mode
function endTimeMode() {
    gameState.timeMode = false;
    clearInterval(gameState.timerInterval);
    elements.timeProgress.style.width = '100%';
    
    if (gameState.score > gameState.highScore) {
        gameState.highScore = gameState.score;
        localStorage.setItem('highScore', gameState.highScore);
        showFeedback('New High Score! 🎉');
    }
}

// Load new digit samples
async function loadNewSamples() {
    try {
        const response = await fetch('/get_samples');
        const samples = await response.json();
        
        elements.samplesGrid.innerHTML = '';
        samples.forEach((sample, index) => {
            const sampleElement = createSampleElement(sample, index);
            elements.samplesGrid.appendChild(sampleElement);
        });
        
        gameState.gamesPlayed++;
        localStorage.setItem('gamesPlayed', gameState.gamesPlayed);
        updateStats();
    } catch (error) {
        console.error('Error loading samples:', error);
        showFeedback('Error loading samples 😕', 'error');
    }
}

// Create a sample element
function createSampleElement(sample, index) {
    const div = document.createElement('div');
    div.className = 'digit-sample';
    div.innerHTML = `<img src="data:image/png;base64,${sample.image}" alt="Digit ${index + 1}">`;
    
    div.addEventListener('click', () => handlePrediction(sample));
    return div;
}

// Handle prediction when a digit is clicked
async function handlePrediction(sample) {
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: sample.image })
        });
        
        const result = await response.json();
        const isCorrect = result.prediction === sample.label;
        
        updateGameState(isCorrect);
        displayPredictionResult(result, sample.label);
        checkAchievements();
    } catch (error) {
        console.error('Error making prediction:', error);
        showFeedback('Error making prediction 😕', 'error');
    }
}

// Update game state after prediction
function updateGameState(isCorrect) {
    if (isCorrect) {
        gameState.score += gameState.streak > 0 ? 10 + gameState.streak : 10;
        gameState.streak++;
        gameState.correctPredictions++;
        showFeedback(`+${10 + (gameState.streak > 0 ? gameState.streak : 0)} points! 🎯`);
    } else {
        gameState.streak = 0;
        showFeedback('Try again! 💪');
    }
    
    gameState.totalPredictions++;
    updateStats();
}

// Display prediction result
function displayPredictionResult(result, actualLabel) {
    const isCorrect = result.prediction === actualLabel;
    elements.currentPrediction.innerHTML = `
        <h3>AI's Guess</h3>
        <div class="prediction-details">
            <p style="color: ${isCorrect ? 'var(--success-color)' : 'var(--error-color)'}">
                ${isCorrect ? '✅ Correct!' : '❌ Incorrect'}
            </p>
            <p>Predicted: ${result.prediction}</p>
            <p>Actual: ${actualLabel}</p>
            <p>Confidence: ${(result.confidence * 100).toFixed(1)}%</p>
        </div>
    `;
}

// Update game statistics
function updateStats() {
    elements.score.textContent = `Score: ${gameState.score}`;
    elements.streak.textContent = `Streak: ${gameState.streak}`;
    elements.highScore.textContent = `Best: ${gameState.highScore}`;
    elements.accuracy.textContent = `${((gameState.correctPredictions / gameState.totalPredictions) * 100 || 0).toFixed(1)}%`;
    elements.gamesPlayed.textContent = gameState.gamesPlayed;
}

// Check and award achievements
function checkAchievements() {
    const newAchievements = [];
    
    if (gameState.correctPredictions === 1) {
        newAchievements.push(achievements.firstCorrect);
    }
    if (gameState.streak === 3) {
        newAchievements.push(achievements.streak3);
    }
    if (gameState.streak === 5) {
        newAchievements.push(achievements.streak5);
    }
    if (gameState.score >= 100 && !gameState.achievements.includes('score100')) {
        newAchievements.push(achievements.score100);
    }
    
    newAchievements.forEach(achievement => {
        if (!gameState.achievements.includes(achievement.id)) {
            gameState.achievements.push(achievement.id);
            showAchievement(achievement);
        }
    });
    
    localStorage.setItem('achievements', JSON.stringify(gameState.achievements));
    displayAchievements();
}

// Display achievements
function displayAchievements() {
    const achievementsList = elements.achievements.querySelector('.achievement-list');
    achievementsList.innerHTML = '';
    
    gameState.achievements.slice(-3).forEach(achievementId => {
        const achievement = achievements[achievementId];
        const div = document.createElement('div');
        div.className = 'achievement';
        div.innerHTML = `${achievement.icon} ${achievement.title}`;
        achievementsList.appendChild(div);
    });
}

// Show achievement notification
function showAchievement(achievement) {
    showFeedback(`Achievement Unlocked: ${achievement.title} ${achievement.icon}`, 'achievement');
}

// Show feedback message
function showFeedback(message, type = 'success') {
    const feedback = document.createElement('div');
    feedback.className = `feedback ${type} pop`;
    feedback.textContent = message;
    
    elements.gameFeedback.appendChild(feedback);
    setTimeout(() => {
        feedback.classList.add('slide-out');
        setTimeout(() => feedback.remove(), 300);
    }, 2000);
}

// Add VS mode functions
function startVsMode() {
    gameState.vsMode.active = true;
    resetVsMode();
    elements.vsModeHeader.style.display = 'flex';
    elements.vsGameContainer.style.display = 'block';
    elements.startVsBtn.style.display = 'block';
    elements.vsResults.style.display = 'block';
    elements.newSamplesBtn.style.display = 'none';
    updateVsStats();
}

function endVsMode() {
    gameState.vsMode.active = false;
    elements.vsModeHeader.style.display = 'none';
    elements.vsGameContainer.style.display = 'none';
    elements.startVsBtn.style.display = 'none';
    elements.vsResults.style.display = 'none';
    elements.newSamplesBtn.style.display = 'block';
    if (gameState.vsMode.timerInterval) {
        clearInterval(gameState.vsMode.timerInterval);
    }
}

function resetVsMode() {
    gameState.vsMode = {
        active: true,
        currentRound: 0,
        totalRounds: 10,
        currentDigit: null,
        humanScore: 0,
        aiScore: 0,
        humanCorrect: 0,
        aiCorrect: 0,
        humanTotalTime: 0,
        aiTotalTime: 0,
        roundStartTime: 0,
        currentTime: 0
    };
}

async function startVsRound() {
    if (gameState.vsMode.currentRound >= gameState.vsMode.totalRounds) {
        showVsFinalResults();
        return;
    }

    try {
        const response = await fetch('/get_samples');
        const samples = await response.json();
        const sample = samples[0];
        
        gameState.vsMode.currentDigit = sample;
        gameState.vsMode.roundStartTime = performance.now();
        
        // Update round number
        elements.currentRound.textContent = gameState.vsMode.currentRound + 1;
        
        // Show the digit
        elements.samplesGrid.innerHTML = `
            <div class="digit-display">
                <img src="data:image/png;base64,${sample.image}" alt="Current Digit">
            </div>
        `;
        
        // Start round timer
        startRoundTimer();
        
        // Enable number pad for human input
        enableNumberPad(true);
        
        // Start AI prediction (with random delay to simulate thinking)
        setTimeout(() => makeAiPrediction(sample), Math.random() * 1000 + 500);
        
    } catch (error) {
        console.error('Error starting VS round:', error);
        showFeedback('Error loading digit 😕', 'error');
    }
}

function enableNumberPad(enabled) {
    document.querySelectorAll('.number-btn').forEach(btn => {
        btn.disabled = !enabled;
    });
}

async function makeAiPrediction(sample) {
    const aiStartTime = performance.now();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: sample.image })
        });
        
        const result = await response.json();
        const aiEndTime = performance.now();
        const aiTime = (aiEndTime - aiStartTime) / 1000;
        
        handleAiGuess(result.prediction, sample.label, aiTime);
    } catch (error) {
        console.error('Error getting AI prediction:', error);
        showFeedback('AI prediction failed 😕', 'error');
    }
}

function handleHumanGuess(guess) {
    const endTime = performance.now();
    const timeTaken = (endTime - gameState.vsMode.roundStartTime) / 1000;
    
    const isCorrect = guess === gameState.vsMode.currentDigit.label;
    gameState.vsMode.humanTotalTime += timeTaken;
    
    if (isCorrect) {
        gameState.vsMode.humanScore += Math.max(10 - Math.floor(timeTaken), 1);
        gameState.vsMode.humanCorrect++;
    }
    
    // Stop the timer
    stopRoundTimer();
    
    // Show feedback popup
    showFeedbackPopup(isCorrect, timeTaken);
    
    enableNumberPad(false);
    updateVsStats();
}

function handleAiGuess(prediction, actualLabel, timeTaken) {
    const isCorrect = prediction === actualLabel;
    gameState.vsMode.aiTotalTime += timeTaken;
    
    if (isCorrect) {
        gameState.vsMode.aiScore += Math.max(10 - Math.floor(timeTaken), 1);
        gameState.vsMode.aiCorrect++;
    }
    
    updateVsStats();
    
    // Show round results
    const roundResults = document.createElement('div');
    roundResults.className = 'stat-row';
    roundResults.innerHTML = `
        <span>Round ${gameState.vsMode.currentRound + 1}</span>
        <span>Human: ${isCorrect ? '✅' : '❌'} (${timeTaken.toFixed(1)}s)</span>
        <span>AI: ${isCorrect ? '✅' : '❌'} (${timeTaken.toFixed(1)}s)</span>
    `;
    elements.vsResults.querySelector('.round-stats').appendChild(roundResults);
}

function updateVsStats() {
    const humanAccuracy = (gameState.vsMode.humanCorrect / Math.max(gameState.vsMode.currentRound, 1) * 100).toFixed(1);
    const aiAccuracy = (gameState.vsMode.aiCorrect / Math.max(gameState.vsMode.currentRound, 1) * 100).toFixed(1);
    
    elements.humanScore.textContent = `${gameState.vsMode.humanScore}pts`;
    elements.humanAccuracy.textContent = `${humanAccuracy}%`;
    elements.humanTime.textContent = `${(gameState.vsMode.humanTotalTime / Math.max(gameState.vsMode.currentRound, 1)).toFixed(1)}s`;
    
    elements.aiScore.textContent = `${gameState.vsMode.aiScore}pts`;
    elements.aiAccuracy.textContent = `${aiAccuracy}%`;
    elements.aiTime.textContent = `${(gameState.vsMode.aiTotalTime / Math.max(gameState.vsMode.currentRound, 1)).toFixed(1)}s`;
}

function showVsFinalResults() {
    const humanWon = gameState.vsMode.humanScore > gameState.vsMode.aiScore;
    const humanAccuracy = (gameState.vsMode.humanCorrect / gameState.vsMode.totalRounds * 100).toFixed(1);
    const aiAccuracy = (gameState.vsMode.aiCorrect / gameState.vsMode.totalRounds * 100).toFixed(1);
    
    elements.vsResults.innerHTML = `
        <h3>Final Results</h3>
        <div class="round-stats">
            <div class="stat-row ${humanWon ? 'winner' : ''}">
                <span>👤 You</span>
                <span>${gameState.vsMode.humanScore}pts</span>
                <span>${humanAccuracy}%</span>
                <span>${(gameState.vsMode.humanTotalTime / gameState.vsMode.totalRounds).toFixed(1)}s avg</span>
            </div>
            <div class="stat-row ${!humanWon ? 'winner' : ''}">
                <span>🤖 AI</span>
                <span>${gameState.vsMode.aiScore}pts</span>
                <span>${aiAccuracy}%</span>
                <span>${(gameState.vsMode.aiTotalTime / gameState.vsMode.totalRounds).toFixed(1)}s avg</span>
            </div>
        </div>
        <div class="winner-announcement">
            ${humanWon ? '🎉 You Win! 🎉' : '🤖 AI Wins! Better luck next time!'}
        </div>
    `;
    
    // Add achievement for beating the AI
    if (humanWon) {
        const achievement = {
            id: 'beatAI',
            icon: '🏆',
            title: 'AI Challenger',
            description: 'Beat the AI in a VS match'
        };
        if (!gameState.achievements.includes('beatAI')) {
            gameState.achievements.push('beatAI');
            showAchievement(achievement);
            localStorage.setItem('achievements', JSON.stringify(gameState.achievements));
        }
    }
    
    elements.startVsBtn.textContent = 'Play Again';
    elements.startVsBtn.onclick = () => {
        resetVsMode();
        elements.vsResults.querySelector('.round-stats').innerHTML = '';
        startVsRound();
    };
}

// Add round timer functions
function startRoundTimer() {
    if (gameState.vsMode.timerInterval) {
        clearInterval(gameState.vsMode.timerInterval);
    }
    
    gameState.vsMode.currentTime = 0;
    updateTimer();
    
    gameState.vsMode.timerInterval = setInterval(() => {
        gameState.vsMode.currentTime += 0.1;
        updateTimer();
    }, 100);
}

function updateTimer() {
    elements.roundTimer.textContent = gameState.vsMode.currentTime.toFixed(1) + 's';
}

function stopRoundTimer() {
    if (gameState.vsMode.timerInterval) {
        clearInterval(gameState.vsMode.timerInterval);
    }
}

// Add feedback popup function
function showFeedbackPopup(isCorrect, timeTaken) {
    const popup = elements.feedbackPopup;
    const content = popup.querySelector('.feedback-content');
    
    popup.className = `feedback-popup ${isCorrect ? 'correct' : 'incorrect'}`;
    content.querySelector('.feedback-message').textContent = isCorrect 
        ? `Correct! (${timeTaken.toFixed(1)}s)` 
        : `Wrong! The correct answer was ${gameState.vsMode.currentDigit.label}`;
    
    popup.classList.add('show');
    
    // Hide popup and start next round after delay
    setTimeout(() => {
        popup.classList.remove('show');
        setTimeout(() => {
            gameState.vsMode.currentRound++;
            startVsRound();
        }, 300);
    }, 1000);
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', initGame);
