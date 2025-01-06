"use client";

import { useChat } from "ai/react";

export default function Page() {
    const { messages, input, handleInputChange, handleSubmit } = useChat({});

    return (
        <main className="relative max-w-screen-md p-4 md:p-6 mx-auto flex min-h-svh !pb-32 md:!pb-40 overflow-y-auto">
            <div className="w-full">
                <p>Test</p>
            </div>
        </main>
    );
}
