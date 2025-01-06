"use client";

import { useChat, Message } from "ai/react";
import cx from "./lib/cx";
import Chat from "@/app/ui/chat";
import Form from "./ui/form";

export default function Page() {
    const { messages, input, handleInputChange, handleSubmit } = useChat({});

    return (
        <main className="relative mx-auto flex min-h-svh max-w-screen-md overflow-y-auto p-4 !pb-32 md:p-6 md:!pb-40">
            <div className="w-full">
                {messages.map((message: Message) => {
                    return <Chat key={message.id} {...message} />;
                })}

                <div
                    className={cx(
                        "fixed inset-x-0 bottom-0 z-10",
                        "flex items-center justify-center",
                        "bg-white",
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
