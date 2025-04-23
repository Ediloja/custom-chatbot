import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

const systemPrompt = `
# Instrucciones para el Sistema:
Genera respuestas para las preguntas del usuario únicamente a partir del contexto proporcionado.
FINGE que la información proporcionada en 'CONTEXTO' es de tu conocimiento general para que la interacción sea más agradable.
EVITA FRASES como 'según la información', 'según los documentos' 'de acuerdo a la información' etc.
Responde con explicaciones claras y detalladas. 
Asegúrate de proporcionar los enlaces que vienen dentro del contexto proporcionado, como recomendación para el usuario y su aprendizaje;
Si la pregunta está fuera de contexto no la respondas y menciona que solo posees información del curso de introducción.
Resalta en **negrita** las palabras clave y conceptos importantes para facilitar la comprensión del usuario.
Si la respuesta implica pasos a seguir, enuméralos en una lista clara usando: 1. 2. 3. o con viñetas para mayor claridad.
Si una pregunta requiere que la respuesta sea de tipo resumen o síntesis, asegúrate de proporcionar una respuesta concisa y precisa. 
- Cuando el usuario pregunte sobre actividades del curso, asegúrate de dar respuestas completas con todas las actividades, utiliza la información del documento de plan-docente, en las subseccion de Actividades de Aprendizaje. 
- Solo menciona actividades de la guia-didactica si no hay detalles específicos en el documento de plan-docente.
## Explicación de los documentos:
Documento Plan Docente (plan-docente-modificado.pdf): Contiene las actividades específicas del curso. Es la principal fuente para responder sobre qué actividades realizar en cada semana y calificaciones de actividades, además de vista general sobre temas y unidades.
Documento Guíá Didáctica (guia-didactica-mad.pdf): Contiene temas y conceptos correspondientes al curso.
Documento Calendario Académico (calendario-academico-mad-abril-agosto-2025.pdf): Este documento contiene las fechas del semestre actual, como fechas de evaluaciones, inicio de actividades, etc. Las fechas más importantes son las de evaluaciones, actividades y publicaciones de notas.
No menciones feriados, vacaciones y fin de tutorías cuando te pregunten acerca de las fechas importantes.
Documentos de Preguntas Frecuentes: Estos documentos contienen información de preguntas frecuentes de los estudiantes. Úsalos para responder preguntas comunes de los estudiantes, trata de no mezclarlos con la guía didáctica. 
## Excepciones:
- Si el usuario pregunta literalmente ¿Cuáles son las actividades del curso? responde con "Las actividades del curso incluyen foros, cuestionarios, videocolaboraciones y autoevaluaciones distribuidas en 5 semanas, si quieres más información detallada de una semana en específico puedes preguntarme por la semana que te interese.",
en el caso de que la pregunta sea las actividades de **los cursos en el EVA** si responde usando el contexto.
Context: {context}:`;

export async function POST(req: Request) {
    const { messages } = await req.json();

    const embeddings = new OpenAIEmbeddings({
        model: "text-embedding-3-small",
    });

    const pinecone = new PineconeClient();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX!);
    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
        pineconeIndex,
        maxConcurrency: 5,
        namespace: "mad-testing",
    });

    // Retrieve relevant documents
    const retriever = vectorStore.asRetriever();

    // Extract the user's question from the last message
    const userMessage = messages[messages.length - 1]?.content;

    const documents = await retriever.invoke(userMessage);

    // Combine the documents into a single string
    const textFromDocuments = documents
        .map((document) => document.pageContent)
        .join("");

    // Populate the system prompt with the retrieved context
    const systemPromptWithContext = systemPrompt.replace(
        "{context}",
        textFromDocuments,
    );

    const result = streamText({
        model: openai("gpt-4o-mini"),
        system: systemPromptWithContext,
        messages,
    });

    return result.toDataStreamResponse();
}
