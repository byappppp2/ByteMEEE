import "dotenv/config";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY, // make sure your key is in .env
});

// === User-defined variables ===
const scanVolume = 250000; // (var)
const flaggedRecords = 327; // (var)
const topViolations = [
  "High-value cross-border transfers",
  "Structuring (multiple small transactions)",
  "Odd-hour transactions"
]; // (var)

// === Prompt generation ===
const prompt = `
You are a compliance analyst. Write a concise fraud detection report for stakeholders.
Use bullet points and short paragraphs.

Details:
- Total transactions scanned: ${scanVolume}
- Number of flagged records: ${flaggedRecords}
- Proportion of flagged records: (calculate automatically)
- Detection methods used: Combination of rule-based detection, logistic regression, and random forest.
- Top 3 rule violations: ${topViolations.join(", ")}

Please produce a clear, professional report that explains findings and insights.
`;

async function generateReport() {
  try {
    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.6,
    });

    console.log("=== Fraud Detection Report ===\n");
    console.log(response.choices[0].message.content);
  } catch (error) {
    console.error("Error generating report:", error);
  }
}

generateReport();