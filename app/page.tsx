"use client";

import { useChat, Message } from "ai/react";
import cx from "./lib/cx";
import Chat from "@/app/ui/chat";
import Form from "@/app/ui/form";
import { INITIAL_QUESTIONS } from "@/app/lib/questions";

export default function Page() {
    const { messages, input, handleInputChange, handleSubmit } = useChat({
        initialMessages: [
            {
                id: "0",
                role: "system",
                content: `**Bienvenido a UTPL Assistant**
                
Tu compañero ideal para resolver dudas sobre los cursos académicos.`,
            },
        ],
    });

    return (
        <main className="relative mx-auto flex min-h-svh max-w-screen-md overflow-y-auto p-4 !pb-32 md:p-6 md:!pb-40">
            <div className="w-full">
                {messages.map((message: Message) => {
                    return <Chat key={message.id} {...message} />;
                })}

                {messages.length === 1 && (
                    <div className="mt-4 grid gap-2 md:mt-6 md:grid-cols-2 md:gap-4">
                        {INITIAL_QUESTIONS.map((message) => {
                            return (
                                <button
                                    key={message.content}
                                    type="button"
                                    className="cursor-pointer select-none rounded-xl border border-gray-200 bg-white p-3 text-left font-normal text-black hover:border-zinc-400 hover:bg-zinc-50 md:px-4 md:py-3"
                                    // onClick={() =>
                                    //     onClickQuestion(message.content)
                                    // }
                                >
                                    {message.content}
                                </button>
                            );
                        })}
                    </div>
                )}

                <div
                    className={cx(
                        "fixed inset-x-0 bottom-0 z-10 flex items-center justify-center bg-white",
                    )}
                >
                    <div className="w-full max-w-screen-md rounded-xl px-4 py-6 md:px-5">
                        <Form
                            onSubmit={handleSubmit}
                            input={input}
                            onChange={handleInputChange}
                        />
                    </div>
                </div>
            </div>
        </main>
    );
}
