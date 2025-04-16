import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "langchain/document";

const documentsPath = "./../../assets/";

async function loadPDFDocuments(documentsPath: string): Promise<Document[]> {
    const directoryLoader = new DirectoryLoader(documentsPath, {
        ".pdf": (filePath: string) => new PDFLoader(filePath),
    });

    return await directoryLoader.load();
}

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

async function processPDFDocuments() {
    const documents = await loadPDFDocuments(documentsPath);

    const splitDocuments = await splitPDFDocuments(documents);
    console.log(splitDocuments.slice(0, 3));
}

processPDFDocuments();
