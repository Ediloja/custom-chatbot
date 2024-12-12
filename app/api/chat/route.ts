import { google } from "@ai-sdk/google";
import { streamText } from "ai";

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
    const { messages } = await req.json();

    const result = streamText({
        model: google("gemini-1.5-pro-latest"),
        system: "You are a helpful assistant.",
        messages,
    });

    return result.toDataStreamResponse();
}
