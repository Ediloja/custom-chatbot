"use client";

import { useRef, useEffect } from "react";
import { useChat, Message } from "ai/react";
import cx from "@/app/lib/cx";
import Chat from "@/app/ui/chat";
import Form from "@/app/ui/form";
import Loading from "@/app/ui/loading";
import Error from "@/app/ui/error";
import Footer from "@/app/ui/footer";
import { INITIAL_QUESTIONS } from "@/app/lib/questions";

export default function Page() {
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const formRef = useRef<HTMLFormElement>(null);

    const {
        messages,
        input,
        handleInputChange,
        handleSubmit,
        setInput,
        isLoading,
        stop,
        error,
        reload,
    } = useChat({
        api: "https://demo-deploy-mu-liard.vercel.app/api/chat",
        maxSteps: 4,
        streamProtocol: "data",
        initialMessages: [
            {
                id: "0",
                role: "system",
                content: `**¡Hola, Jaguar UTPL!**

Bienvenido a tu compañero ideal para conquistar tus metas académicas.`,
            },
        ],

        onError: (error) => console.log(`An error occurred ${error}`),
    });

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

    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView();
        }
    }, [messages]);

    return (
        <main className="relative mx-auto flex min-h-svh max-w-screen-md overflow-y-auto p-4 !pb-32 md:p-6 md:!pb-40">
            <div className="w-full">
                {messages.map((message: Message) => {
                    return <Chat key={message.id} {...message} />;
                })}

                {isLoading && <Loading />}

                {error && <Error />}

                {messages.length === 1 && (
                    <div className="mt-4 grid gap-2 md:mt-6 md:grid-cols-2 md:gap-4">
                        {INITIAL_QUESTIONS.map((message) => {
                            return (
                                <button
                                    key={message.content}
                                    type="button"
                                    className="cursor-pointer select-none rounded-xl border border-gray-200 bg-white p-3 text-left font-normal text-black hover:border-zinc-400 hover:bg-zinc-50 md:px-4 md:py-3"
                                    onClick={() =>
                                        handleClickInitialQuestion(
                                            message.content,
                                        )
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
