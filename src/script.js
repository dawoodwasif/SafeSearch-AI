// Function to dynamically generate decoy queries
function generateDecoyQueries(realQuery, k) {
    const commonWords = [
        "how", "what", "when", "where", "who", "best", "top", "guide", "tips", "learn", "benefits", "history", "facts", "reasons", "why", "how much", "does", "is", "can", "should", "about", "latest", "trending"
    ];

    const randomTopics = [
        "technology", "health", "sports", "movies", "education", "finance", "travel", "music", "cooking", "gardening", "fitness", "science"
    ];

    const randomModifiers = [
        "today", "this year", "in 2024", "for beginners", "step by step", "quick tips", "in my city", "best practices", "for experts", "in the world", "locally"
    ];

    const generatedQueries = [];

    // Helper function to capitalize the first letter of a sentence
    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }

    // Generate k-1 decoy queries
    for (let i = 0; i < k - 1; i++) {
        // Pick a random common word, random topic, and random modifier
        const randomCommonWord = commonWords[Math.floor(Math.random() * commonWords.length)];
        const randomTopic = randomTopics[Math.floor(Math.random() * randomTopics.length)];
        const randomModifier = randomModifiers[Math.floor(Math.random() * randomModifiers.length)];

        // Form a decoy query by combining parts randomly
        const decoyQuery = `${capitalizeFirstLetter(randomCommonWord)} about ${randomTopic} ${randomModifier}`;
        generatedQueries.push(decoyQuery);
    }

    // Return the real query + decoys (in random order)
    const allQueries = [...generatedQueries, realQuery].sort(() => 0.5 - Math.random());

    return allQueries;
}

// Example of using the function to generate decoy queries
function sendQueries(realQuery) {
    const k = 5;  // Define k value for k-anonymity
    const queries = generateDecoyQueries(realQuery, k);
    
    console.log("Generated Queries: ", queries);
    // Send these queries to the search engine (integrate with the rest of your app here)
    queries.forEach(query => {
        console.log(`Query: ${query}`); // Example of where you'd send to API
    });
}

// Example: Call this when a user inputs a query
const realQuery = "Who won the Super Bowl this year?";
sendQueries(realQuery);
