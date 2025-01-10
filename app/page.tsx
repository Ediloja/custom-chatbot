"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useChat, Message } from "ai/react";
import cx from "@/app/lib/cx";
import Chat from "@/app/ui/chat";
import Form from "@/app/ui/form";
import Loading from "@/app/ui/loading";
import Footer from "@/app/ui/footer";
import { INITIAL_QUESTIONS } from "@/app/lib/questions";

export default function Page() {
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const formRef = useRef<HTMLFormElement>(null);

    const [streaming, setStreaming] = useState(false);

    const { messages, input, handleInputChange, handleSubmit, setInput } =
        useChat({
            initialMessages: [
                {
                    id: "0",
                    role: "system",
                    content: `**Bienvenido a UTPL Assistant**
                
Tu compañero ideal para resolver dudas sobre los cursos académicos.`,
                },
            ],

            onResponse: () => {
                setStreaming(false);
            },
        });

    function handleClickQuestion(value: string) {
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

    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView();
        }
    }, [messages]);

    const onSubmit = useCallback(
        (e: React.FormEvent<HTMLFormElement>) => {
            e.preventDefault();
            handleSubmit(e);
            setStreaming(true);
        },
        [handleSubmit],
    );

    return (
        <main className="relative mx-auto flex min-h-svh max-w-screen-md overflow-y-auto p-4 !pb-32 md:p-6 md:!pb-40">
            <div className="w-full">
                {messages.map((message: Message) => {
                    return <Chat key={message.id} {...message} />;
                })}

                {streaming && <Loading />}

                {messages.length === 1 && (
                    <div className="mt-4 grid gap-2 md:mt-6 md:grid-cols-2 md:gap-4">
                        {INITIAL_QUESTIONS.map((message) => {
                            return (
                                <button
                                    key={message.content}
                                    type="button"
                                    className="cursor-pointer select-none rounded-xl border border-gray-200 bg-white p-3 text-left font-normal text-black hover:border-zinc-400 hover:bg-zinc-50 md:px-4 md:py-3"
                                    onClick={() =>
                                        handleClickQuestion(message.content)
                                    }
                                >
                                    {message.content}
                                </button>
                            );
                        })}
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
                            onSubmit={onSubmit}
                            onChange={handleInputChange}
                            disabled={streaming}
                        />
                        <Footer />
                    </div>
                </div>
            </div>
        </main>
    );
}
