import { generateText } from "ai"

export async function POST(request: Request) {
  try {
    const { transactions } = await request.json()

    const { text } = await generateText({
      model: "openai/gpt-4o-mini",
      prompt: `You are a financial compliance analyst. Generate a comprehensive compliance summary report for the following flagged transactions.

Transactions Data:
${JSON.stringify(transactions, null, 2)}

Please provide:
1. Executive Summary
2. Risk Assessment Overview
3. Detailed Analysis by Flag Type
4. Recommendations for Further Investigation
5. Compliance Actions Required

Format the report professionally with clear sections and actionable insights.`,
    })

    return Response.json({ report: text })
  } catch (error) {
    console.error("Error generating report:", error)
    return Response.json({ error: "Failed to generate report" }, { status: 500 })
  }
}