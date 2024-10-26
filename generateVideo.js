import fetch from 'node-fetch';
import { translateText } from './translate.js';

async function createVideoFromScript(scriptText) {
  const heygenApiKey = 'YOUR_HEYGEN_API_KEY'; // Replace with your Heygen API key
  const heygenUrl = 'https://api.heygen.com/v1/video/generate'; // Hypothetical endpoint
  const templateId = 'YOUR_TEMPLATE_ID'; // Replace with your template ID

  const response = await fetch(heygenUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${heygenApiKey}`
    },
    body: JSON.stringify({
      template_id: templateId,
      script: scriptText,
      language: 'hi' // Hindi, based on the translated script
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error(`Error creating video: ${response.statusText} - ${errorText}`);
    return;
  }

  const data = await response.json();
  console.log(`Video created successfully! Video URL: ${data.video_url}`);
}

async function main() {
  const englishText = 'Hello, how are you?';
  const translatedText = await translateText(englishText);
  if (translatedText) {
    await createVideoFromScript(translatedText);
  }
}

main();
