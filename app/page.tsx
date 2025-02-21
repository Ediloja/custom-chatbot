"use client";

import { useRef, useEffect, useState } from "react";
import { useChat, Message } from "@ai-sdk/react";
import cx from "@/app/lib/cx";
import Chat from "@/app/ui/chat";
import Form from "@/app/ui/form";
import Loading from "@/app/ui/loading";
import Error from "@/app/ui/error";
import Footer from "@/app/ui/footer";
import { INITIAL_QUESTIONS } from "@/app/lib/questions";

export default function Page() {
    // State to store the chat history from localStorage
    const [persistedMessages, setPersistedMessages] = useState<Message[]>([]);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const formRef = useRef<HTMLFormElement>(null);

    // On component mount, load the chat history from localStorage if available
    useEffect(() => {
        const stored = localStorage.getItem("chatHistory");

        if (stored) {
            setPersistedMessages(JSON.parse(stored));
        }
    }, []);

    // Initialize useChat with persisted messages or fallback to the default welcome message
    const {
        messages,
        input,
        handleInputChange,
        handleSubmit,
        status,
        setInput,
        isLoading,
        stop,
        error,
        reload,
    } = useChat({
        api: "http://127.0.0.1:8000/api/chat",
        maxSteps: 4,
        streamProtocol: "data",
        initialMessages:
            persistedMessages.length > 0
                ? persistedMessages
                : [
                      {
                          id: "0",
                          role: "system",
                          content: `**¡Hola! Soy SophiaUTPL**

Tu asistente virtual para conquistar tus metas académicas.`,
                      },
                  ],
        onError: (error) => console.log(`An error occurred ${error}`),
    });

    // Save the current chat history to localStorage whenever 'messages' changes.
    useEffect(() => {
        localStorage.setItem("chatHistory", JSON.stringify(messages));
    }, [messages]);

    // When an initial question is clicked, set the input and dispatch a submit event.
    function handleClickInitialQuestion(value: string) {
        setInput(value);
        setTimeout(() => {
            formRef.current?.dispatchEvent(
                new Event("submit", {
                    cancelable: true,
                    bubbles: true,
                }),
            );
        }, 1);
    }

    // Auto-scroll to the bottom whenever messages update.
    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView();
        }
    }, [messages]);

    return (
        <main className="relative mx-auto flex min-h-svh max-w-screen-md overflow-y-auto p-4 !pb-32 md:p-6 md:!pb-40">
            <div className="w-full">
                {messages.map((message: Message) => (
                    <Chat key={message.id} {...message} />
                ))}

                {status === "submitted" && <Loading />}

                {error && <Error />}

                {messages.length === 1 && (
                    <div className="mt-4 grid gap-2 md:mt-6 md:grid-cols-2 md:gap-4">
                        {INITIAL_QUESTIONS.map((message) => (
                            <button
                                key={message.content}
                                type="button"
                                className="cursor-pointer select-none rounded-xl border border-gray-200 bg-white p-3 text-left font-normal text-black hover:border-zinc-400 hover:bg-zinc-50 md:px-4 md:py-3"
                                onClick={() =>
                                    handleClickInitialQuestion(message.content)
                                }
                            >
                                {message.content}
                            </button>
                        ))}
                    </div>
                )}

                <div ref={messagesEndRef}></div>

                <div
                    className={cx(
                        "fixed inset-x-0 bottom-0 z-10 flex items-center justify-center bg-white",
                    )}
                >
                    <div className="w-full max-w-screen-md rounded-xl px-4 py-6 md:px-5">
                        <Form
                            ref={formRef}
                            input={input}
                            onSubmit={handleSubmit}
                            onChange={handleInputChange}
                            isLoading={isLoading}
                            stop={stop}
                            error={error}
                            reload={reload}
                        />
                        <Footer />
                    </div>
                </div>
            </div>
        </main>
    );
}
