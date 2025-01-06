"use client";

import { useChat, Message } from "ai/react";
import Chat from "./ui/Chat";

export default function Page() {
    const { messages, input, handleInputChange, handleSubmit } = useChat({});

    return (
        <main className="relative mx-auto flex min-h-svh max-w-screen-md overflow-y-auto p-4 !pb-32 md:p-6 md:!pb-40">
            <div className="w-full">
                {messages.map((message: Message) => {
                    return <Chat key={message.id} {...message} />;
                })}

                <form onSubmit={handleSubmit}>
                    <input
                        name="prompt"
                        value={input}
                        onChange={handleInputChange}
                    />
                    <button type="submit">Submit</button>
                </form>
            </div>
        </main>
    );
}
