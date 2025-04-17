import "dotenv/config";
import path from "path";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";

// Verificar que existan todas las variables de entorno requeridas
function validateRequiredEnvironmentVariables(): void {
    const requiredEnvironmentVariables: string[] = [
        "PINECONE_API_KEY",
        "PINECONE_INDEX",
        "OPENAI_API_KEY",
    ];

    const missingEnvironmentVariables: string[] =
        requiredEnvironmentVariables.filter(
            (environmentVariable) => !process.env[environmentVariable],
        );

    if (missingEnvironmentVariables.length > 0) {
        throw new Error(
            `Missing required environment variables: ${missingEnvironmentVariables.join(", ")}`,
        );
    }
}

// Ruta absoluta de los documentos PDF (robusta ante la ubicación de ejecución)
const documentsPath: string = path.resolve(process.cwd(), "../../assets");

// Cargar los documentos PDF desde el directorio especificado
async function loadPDFDocuments(documentsPath: string): Promise<Document[]> {
    const directoryLoader = new DirectoryLoader(documentsPath, {
        ".pdf": (filePath: string) => new PDFLoader(filePath),
    });

    return await directoryLoader.load();
}

// Dividir los documentos en chunks para el procesamiento
async function splitPDFDocuments(
    documents: Document[],
    chunkSize: number = 1000,
    chunkOverlap: number = 200,
): Promise<Document[]> {
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize,
        chunkOverlap,
    });

    return await textSplitter.splitDocuments(documents);
}

// Procesar los documentos PDF
async function processPDFDocuments(): Promise<Document[]> {
    try {
        const documents = await loadPDFDocuments(documentsPath);
        const splitDocuments = await splitPDFDocuments(documents);

        console.log("Sample split documents:", splitDocuments.slice(0, 3));

        return splitDocuments;
    } catch (error) {
        console.error("Error processing PDF documents:", error);

        throw error;
    }
}

// Modelo de embeddings de OpenAI
function createEmbeddings(): OpenAIEmbeddings {
    return new OpenAIEmbeddings({
        model: "text-embedding-3-small",
    });
}

// Conexión con Pinecone
function connectToPinecone(): ReturnType<PineconeClient["Index"]> {
    if (!process.env.PINECONE_INDEX) {
        throw new Error(
            "PINECONE_INDEX is not defined in environment variables.",
        );
    }

    const pinecone = new PineconeClient();

    return pinecone.Index(process.env.PINECONE_INDEX!);
}

// Función principal para ejecutar el pipeline RAG
async function runRAGPipeline(): Promise<void> {
    try {
        validateRequiredEnvironmentVariables();

        const embeddings = createEmbeddings();
        const pineconeIndex = connectToPinecone();
        const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
            pineconeIndex,
            maxConcurrency: 5,
            namespace: "rag-chatbot-nextjs",
        });
        const documents = await processPDFDocuments();

        await vectorStore.addDocuments(documents);

        console.log(
            "Vector store initialized and documents added successfully.",
        );
    } catch (error) {
        console.error("RAG pipeline initialization failed:", error);
    }
}

runRAGPipeline();
