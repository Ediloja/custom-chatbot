import { Message } from "ai/react";
import cx from "../lib/cx";
import Markdown from "markdown-to-jsx";

export default function Chat({ content, role }: Message) {
    const isUser = role === "user";

    return (
        <article
            className={cx(
                "mb-4 flex items-start gap-4 rounded-2xl px-10 py-5",
                isUser ? "" : "bg-utpl-primary text-white",
            )}
        >
            <Markdown
                className={cx(
                    "space-y-4 py-2 leading-normal md:py-2",
                    isUser ? "font-semibold" : "",
                )}
                options={{
                    overrides: {
                        ol: ({ children }) => (
                            <ol className="list-decimal">{children}</ol>
                        ),
                        ul: ({ children }) => (
                            <ol className="list-disc">{children}</ol>
                        ),
                    },
                }}
            >
                {content}
            </Markdown>
        </article>
    );
}
