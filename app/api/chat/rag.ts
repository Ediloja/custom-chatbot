import "dotenv/config";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";

// Path de los documentos PDF
const documentsPath = "./../../assets/";

// Carga los documentos PDF desde el directorio especificado
async function loadPDFDocuments(documentsPath: string): Promise<Document[]> {
    const directoryLoader = new DirectoryLoader(documentsPath, {
        ".pdf": (filePath: string) => new PDFLoader(filePath),
    });

    return await directoryLoader.load();
}

// Divide los documentos en chunks para procesamiento
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

// Procesa los documentos PDF
async function processPDFDocuments() {
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

// Crea los embeddings de OpenAI
function createEmbeddings() {
    return new OpenAIEmbeddings({
        model: "text-embedding-3-small",
    });
}

// Conecta con Pinecone usando la variable de entorno
function connectToPinecone() {
    if (!process.env.PINECONE_INDEX) {
        throw new Error(
            "PINECONE_INDEX is not defined in environment variables.",
        );
    }

    const pinecone = new PineconeClient();

    return pinecone.Index(process.env.PINECONE_INDEX!);
}

// Inicializa la conexión y almacenamiento en Pinecone
async function initializePinecone() {
    try {
        if (
            !process.env.PINECONE_API_KEY ||
            !process.env.PINECONE_INDEX ||
            !process.env.OPENAI_API_KEY
        ) {
            throw new Error("Missing required environment variables.");
        }

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
        console.error("Initialization failed:", error);
    }
}

initializePinecone();
