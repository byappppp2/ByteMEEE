import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

const scanVolume = 250000; // (var)
const flaggedRecords = 327; // (var)
const topViolations = [
  "High-value cross-border transfers",
  "Structuring (multiple small transactions)",
  "Odd-hour transactions"
]; // (var)

export async function POST() {
  try {
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

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.6,
    });

    return Response.json({
      report: response.choices[0].message.content,
    });
  } catch (error) {
    console.error("Error generating report:", error);
    return Response.json({ error: "Failed to generate report" }, { status: 500 });
  }
}
