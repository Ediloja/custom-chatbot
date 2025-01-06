import { Message } from "ai/react";
import cx from "../lib/cx";

export default function Chat({ content, role }: Message) {
    const isUser = role === "user";

    return (
        <article
            className={cx(
                "mb-4 flex items-start gap-4 rounded-2xl p-4 md:p-5",
                isUser ? "" : "bg-emerald-50",
            )}
        >
            {content}
        </article>
    );
}
