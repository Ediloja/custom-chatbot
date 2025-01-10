import { ComponentProps } from "react";
import cx from "../lib/cx";
import { IconArrowBack } from "@tabler/icons-react";

export default function Form({
    onSubmit,
    input,
    onChange,
    disabled,
    ref,
}: any) {
    return (
        <form
            onSubmit={onSubmit}
            className="relative m-auto flex items-center justify-center gap-4"
            ref={ref}
        >
            <input
                name="prompt"
                type="text"
                value={input}
                placeholder="Escribe cualquier duda aquí"
                required
                className={cx(
                    "h-10 flex-1 rounded-xl pl-4 pr-12 transition md:h-12",
                    "border border-gray-400 text-base text-black",
                    "disabled:bg-gray-100",
                )}
                onChange={onChange}
                disabled={disabled}
            />

            <button
                type="submit"
                tabIndex={-1}
                className={cx(
                    "absolute right-3 top-1/2 -translate-y-1/2",
                    "opacity-50",
                )}
                disabled={disabled}
            >
                <IconArrowBack stroke={1.5} color="black" />
            </button>
        </form>
    );
}
