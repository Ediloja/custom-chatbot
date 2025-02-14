import { Message } from "ai/react";
import cx from "../lib/cx";
import Markdown from "markdown-to-jsx";
import { IconUserCircle } from "@tabler/icons-react";

export default function Chat({ content, role }: Message) {
    const isUser = role === "user";

    return (
        <article
            className={cx(
                "mb-2 flex items-start gap-4 rounded-2xl p-4 md:p-5",
                isUser ? "text-black" : "bg-utpl-primary text-white",
            )}
        >
            <Avatar isUser={isUser} />
            <Markdown
                className={cx(
                    "space-y-4 py-2 md:leading-7",
                    isUser ? "font-semibold" : "",
                )}
                options={{
                    overrides: {
                        a: {
                            component: ({ href, children }) => (
                                <a
                                    href={href}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-blue-400 underline hover:text-blue-600"
                                >
                                    {children}
                                </a>
                            ),
                        },

                        ol: {
                            component: ({ children }) => (
                                <ol className="list-decimal pl-5">
                                    {children}
                                </ol>
                            ),
                        },

                        ul: {
                            component: ({ children }) => (
                                <ul className="list-disc pl-5">{children}</ul>
                            ),
                        },
                    },
                }}
            >
                {content}
            </Markdown>
        </article>
    );
}

export function Avatar({
    isUser = false,
    className,
}: {
    isUser?: boolean;
    className?: string;
}) {
    return (
        <div
            className={cx(
                "flex size-8 shrink-0 items-center justify-center rounded-full",
                className,
            )}
        >
            {isUser ? (
                <IconUserCircle stroke={1} />
            ) : (
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    fill="none"
                    viewBox="0 0 16 16"
                >
                    <path
                        fill="white"
                        fillRule="evenodd"
                        d="M12.707 2 14 3.293V5h1v1h-1v1.5h1v1h-1V10h1v1h-1v1.707L12.707 14H11v1h-1v-1H8.5v1h-1v-1H6v1H5v-1H3.293L2 12.707V11H1v-1h1V8.5H1v-1h1V6H1V5h1V3.293L3.293 2H5V1h1v1h1.5V1h1v1H10V1h1v1h1.707ZM8.57 5.73 8 4l-.57 1.73a2.667 2.667 0 0 1-1.7 1.7L4 8l1.73.57a2.667 2.667 0 0 1 1.7 1.7L8 12l.57-1.73a2.667 2.667 0 0 1 1.7-1.7L12 8l-1.73-.57a2.667 2.667 0 0 1-1.7-1.7Z"
                        clipRule="evenodd"
                    />
                </svg>
            )}
        </div>
    );
}
