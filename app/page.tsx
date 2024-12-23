"use client";

import { useChat } from "ai/react";
import Form from "./ui/Form";
import Chat from "./ui/Chat";

export default function Page() {
    const { messages, input, handleInputChange, handleSubmit } = useChat({});

    return (
        <>
            <main className="relative max-w-screen-md p-4 md:p-6 mx-auto flex min-h-svh !pb-32 md:!pb-40 overflow-y-auto">
                {/* <Chat /> */}
                <div className="w-full max-w-screen-md rounded-xl px-4 md:px-5 py-6">
                    <Form />
                </div>
            </main>
        </>
    );
}
