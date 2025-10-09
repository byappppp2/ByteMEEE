import { NextResponse } from "next/server";
import OpenAI from "openai";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const transactions = body.transactions;

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) throw new Error("OPENAI_API_KEY is not defined");

    const openai = new OpenAI()

    const response = await openai.responses.create({
      model: "gpt-4o-mini",
      input: [
        {
          role: "system",
          content: "You are a financial compliance analyst."
        },
        {
          role: "user",
          content: `Analyze the following transactions:\n${JSON.stringify(transactions, null, 2)}`
        }
      ]
    });

      const reportText = response.output_text ?? "";

    return NextResponse.json({ report: reportText });
  } catch (error) {
    console.error("Error generating report:", error);
    return NextResponse.json({ error: (error as Error).message }, { status: 500 });
  }
}

