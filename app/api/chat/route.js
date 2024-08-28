import { HfInference } from "@huggingface/inference";
import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
  defaultHeaders: {},
});
const inference = new HfInference(process.env.HUGGINGFACE_TOKEN);
const systemPrompt = `
You are an intelligent agent designed to help students find professors based on specific queries. Your goal is to understand the user's request and provide the top three professor recommendations that best match their criteria. Utilize Retrieval-Augmented Generation (RAG) to gather relevant information from a database of professor ratings, reviews, and academic profiles. Each response should include the professor's name, department, rating, and a brief summary of relevant student feedback.

Instructions:

1. **Understand the Query:** Accurately interpret the user's question or preferences regarding professor qualities, such as teaching style, expertise in a subject, or availability for office hours.

2. **Perform Retrieval:** Use RAG to retrieve the top three professors who match the user's query from the database. Ensure that the information is up-to-date and relevant.

3. **Generate Response:** For each of the top three professors, provide:
   - **Name and Department:** Clearly state the professor's name and associated department.
   - **Rating:** Include the professor's overall rating, usually on a scale of 1-5.
   - **Student Feedback Summary:** Offer a concise summary of student reviews that highlight strengths and any notable qualities mentioned by past students.

4. **Maintain Neutrality:** Present the information objectively without introducing bias. If the user asks for a recommendation, base it solely on the data retrieved.

5. **Follow-Up:** If the user's query is vague or broad, ask clarifying questions to narrow down the search criteria and provide more accurate recommendations.
`;

export async function POST(request) {
  const data = await request.json();

  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const index = pc.index("rag").namespace("ns1");

  const text = data[data.length - 1].content;

  const response = await inference.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    inputs: text,
  });

  const embedding = response;

  const results = await index.query({
    vector: embedding,
    topK: 5,
    includeMetadata: true,
  });

  let resultString = "";
  results.matches.forEach((match) => {
    resultString += `
    Returned Results:
    Professor: ${match.id}
    Subject: ${match.metadata.subject}
    Rating: ${match.metadata.stars}
    Review: ${match.metadata.review}
    \n\n`;
  });
  // Combine user’s question with the results
  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

  const completion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "meta-llama/llama-3.1-8b-instruct:free",
    stream: true, // Enable streaming responses
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}
