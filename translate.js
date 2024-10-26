// Import the Translation client library
const {TranslationServiceClient} = require('@google-cloud/translate').v3;

// Initialize the Translation client
const translationClient = new TranslationServiceClient();

async function translateText(text, targetLanguageCode) {
  const projectId = 'YOUR_PROJECT_ID'; // Replace with your Google Cloud project ID
  const location = 'global'; // Default location for translation service

  // Construct request
  const request = {
    parent: `projects/${projectId}/locations/${location}`,
    contents: [text],
    mimeType: 'text/plain', // Can be 'text/html' for HTML content
    targetLanguageCode: targetLanguageCode,
  };

  try {
    // Make the translation request
    const [response] = await translationClient.translateText(request);

    // Display the translation for each input text
    response.translations.forEach((translation, i) => {
      console.log(`Translated Text: ${translation.translatedText}`);
    });
  } catch (error) {
    console.error(`Error during translation: ${error.message}`);
  }
}

// Call the function with example parameters
translateText('Hello, how are you?', 'es'); // 'es' for Spanish
